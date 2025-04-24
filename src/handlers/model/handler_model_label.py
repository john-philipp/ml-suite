import os
import shutil
import threading

from src.path_helper import PathHelper
from src.parsers.enums import DataFormatType
from src.parsers.interfaces import _Args
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler
from src.methods import try_read_env_var, run_bash, round_float, get_dir_size, \
    count_labels, read_screenshot_config, screenshotter, kill_process
from src.path_handler import PathHandler
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerModelLabel(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.dataset_index = args.dataset_index
            self.data_format = args.data_format
            self.write_screenshot_config = args.write_screenshot_config
            self.take_screenshot = args.take_screenshot
            self.kill_after = args.kill_after

    def handle(self):
        args: HandlerModelLabel.Args = self.args

        if args.data_format == DataFormatType.KITTI:
            self.open_point_labeler("_generated/02-datasets", args.dataset_index, args)
            return

        raise NotImplementedError()

    @classmethod
    def open_point_labeler(
            cls, path, dataset_index, args, file_log=None):

        labeler_path = try_read_env_var("POINT_LABELER_PATH")

        dataset_path_handler = PathHandler(path)
        dataset_path = dataset_path_handler.get_path(dataset_index)

        close_file_log = False
        if not file_log:
            file_log = FileLogger(dataset_path)
            file_log.log(f"Handler: {cls.__name__}")
            file_log.log(f"Dataset: {file_log.link(dataset_path)}")
            close_file_log = True

        screenshot_config_path = f"{dataset_path}/screenshot-config.yml"
        if args.write_screenshot_config:
            file_log.log(f"Writing screenshot config here: {screenshot_config_path}")
            with open(screenshot_config_path, "w") as f:
                f.write(args.screenshot_config)

        screenshot_thread = None
        if args.take_screenshot:
            zoom, up, right = read_screenshot_config(args.screenshot_config, screenshot_config_path, file_log.log)
            screenshot_thread = threading.Thread(target=screenshotter, args=(file_log, zoom, up, right))
            screenshot_thread.start()

        sequence_path = f"{dataset_path}/dataset/sequences/.full"

        try:
            if args.kill_after:
                file_log.log(f"Will kill process in {args.kill_after}s...")
                kill_thread = threading.Thread(target=kill_process, args=("labeler", args.kill_after, True, file_log.log))
                kill_thread.start()
        except AttributeError:
            pass

        run_bash(
            f"./labeler --open-dir {os.getcwd()}/{sequence_path}",
            cwd=f"{labeler_path}/bin")

        file_log.log("Counting labels...")
        label_counts = count_labels(dataset_path)
        labeled_all = sum([y for x, y in label_counts.items() if x != 0])

        unlabeled = 0
        if 0 in label_counts:
            unlabeled = label_counts[0]
        labeled_pec = round_float(labeled_all / (labeled_all + unlabeled), 2)

        file_log.log(f"Done. Labeled: {labeled_pec}")

        performed_action = dict(
            timestamp=file_log.timestamp,
            type="label",
            dataset=file_log.link(dataset_path),
            dataset_size=get_dir_size(dataset_path),
            logs=file_log.link(file_log.get_log_dir()),
            args=args.__dict__,
            label_counts=label_counts,
            labeled=labeled_pec,
        )

        file_log.add_infos(root=True, append=True, performed_actions=performed_action)
        file_log.add_infos(local=True, performed_actions=[performed_action])
        file_log.log(f"Written to: {FileLogger.link(dataset_path)}")

        if screenshot_thread:
            screenshot_thread.join(timeout=5)
            labeler_path = try_read_env_var("POINT_LABELER_PATH")
            screenshot_src = f"{labeler_path}/bin/screenshot.png"
            prediction_name = os.path.basename(dataset_path)
            screenshot_dst = PathHelper().generated().screenshots().push(f"{prediction_name}.png").path()
            shutil.copyfile(screenshot_src, screenshot_dst)
            file_log.log(f"Screenshot saved here: {screenshot_dst}")

        if close_file_log:
            file_log.close()
