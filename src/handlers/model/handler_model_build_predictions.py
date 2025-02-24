import os
import shutil
import uuid

from tqdm import tqdm

from src.path_helper import PathHelper
from src.parsers.enums import ArchitectureType, DataFormatType, ModelType
from src.parsers.interfaces import _Args
from src.file_helpers import rm, cp, mk, pj
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler
from src.methods import mark_path, cp_markers
from src.path_handler import PathHandler
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerModelBuildPredictions(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.training_index = args.training_index
            self.dataset_index = args.dataset_index

            self.data_format = args.data_format
            self.model = args.model
            self.arch = args.arch

    def handle(self):
        args: HandlerModelBuildPredictions.Args = self.args

        if args.arch == ArchitectureType.OPEN3D_ML:
            if args.data_format == DataFormatType.KITTI:
                if args.model == ModelType.RANDLANET:
                    self._predict_open3d_ml_kitti_randlanet(args)
                    return

        raise NotImplementedError()

    @classmethod
    def _predict_open3d_ml_kitti_randlanet(cls, args: Args):

        config_name = "config.yml"
        predictions_path_handler = PathHandler("_generated/04-predictions")
        prediction_uid = uuid.uuid4().hex
        prediction_path = predictions_path_handler.get_next_path(prediction_uid[:4])
        mk(prediction_path)
        file_log = FileLogger(prediction_path)
        file_log.log(f"Handler: {cls.__name__}")
        file_log.add_infos(local=True, root=True, args=args.__dict__)

        dataset_path_handler = PathHandler("_generated/02-datasets")
        dataset_path = dataset_path_handler.get_path(args.dataset_index)
        cp_markers(dataset_path, prediction_path)
        config_file = f"{prediction_path}/{config_name}"
        shutil.copyfile(f"{dataset_path}/{config_name}", config_file)

        try:
            screenshot_config_name = "screenshot-config.yml"
            screenshot_config_file = f"{prediction_path}/{screenshot_config_name}"
            shutil.copyfile(f"{dataset_path}/{screenshot_config_name}", screenshot_config_file)
            file_log.log("Copied screenshot config.")
        except FileNotFoundError:
            file_log.log("No screenshot config found.")

        file_log.log(f"Dataset: {file_log.link(dataset_path)}")
        file_log.keep_file(config_file)

        training_path_handler = PathHandler("_generated/03-trainings")
        training_path = training_path_handler.get_path(args.training_index)
        cp_markers(training_path, prediction_path)
        mark_path(prediction_path, "training", training_path.split("_")[-1])

        file_log.log(f"Training: {training_path}")
        training_log = os.path.basename(PathHelper(training_path, "log").latest().path())
        file_log.add_infos(local=True, root=True, training_log=training_log, training_path=training_path)

        test_seqs_dir = f"{training_path}/randlanet/test/sequences"
        if not os.path.isdir(test_seqs_dir) or os.listdir(test_seqs_dir) == 0:
            file_log.log(f"Not yet tested.")
            file_log.close()
            raise AssertionError("Did you test, yet?")

        test_seqs = os.listdir(test_seqs_dir)
        test_seqs.sort()

        # For each sequence.
        pred_seqs_dir = f"{prediction_path}/dataset/sequences"
        file_log.log(f"Copying from sequence: {file_log.link(dataset_path)}")
        progress = tqdm(desc=f"Copying", total=len(test_seqs))

        for test_seq in test_seqs:
            # Copy from kitti. Original sequence.
            orig_seq_dir = f"{dataset_path}/dataset/sequences/{test_seq}"
            test_seq_dir = f"{test_seqs_dir}/{test_seq}"
            pred_seq_dir = f"{pred_seqs_dir}/{test_seq}"
            mk(pred_seqs_dir)

            if os.path.exists(pred_seq_dir):
                rm(pred_seq_dir)
            cp(orig_seq_dir, pred_seq_dir)

            # Remove labels dir.
            # Copy predictions dir into sequence as labels dir.
            src_path = f"{test_seq_dir}/predictions"
            dst_path = f"{pred_seq_dir}/labels"
            rm(dst_path)
            cp(src_path, dst_path)
            progress.update(1)

        progress.close()

        # Now combine to full sequence.
        pred_full_dir = f"{pred_seqs_dir}/.full"
        rm(pred_full_dir)
        full_labels_dir = f"{pred_full_dir}/labels"
        full_bins_dir = f"{pred_full_dir}/velodyne"
        mk(full_labels_dir)
        mk(full_bins_dir)

        full_poses_f = open(f"{pred_full_dir}/poses.txt", "w")
        full_times_f = open(f"{pred_full_dir}/times.txt", "w")

        offset = 0
        time_offset = 0
        file_log.log(f"Copying from predictions: {file_log.link(prediction_path)}")
        progress = tqdm(desc=f"Copying", total=len(test_seqs))
        for test_seq in test_seqs:
            pred_seq_dir = f"{pred_seqs_dir}/{test_seq}"
            pred_labels_dir = f"{pred_seq_dir}/labels"
            pred_bins_dir = f"{pred_seq_dir}/velodyne"
            pred_poses_f = open(f"{pred_seq_dir}/poses.txt")
            pred_times_f = open(f"{pred_seq_dir}/times.txt")

            label_files = os.listdir(pred_labels_dir)
            label_files.sort()
            bin_files = os.listdir(pred_bins_dir)
            bin_files.sort()

            for i in range(0, len(label_files)):
                j = i + offset
                shutil.copyfile(pj(
                    pred_labels_dir, label_files[i]), pj(full_labels_dir, f"{j:06}.label"))
                shutil.copyfile(pj(
                    pred_bins_dir, bin_files[i]), pj(full_bins_dir, f"{j:06}.bin"))
            offset += len(label_files)

            full_poses_f.writelines(pred_poses_f.readlines())
            times = [float(x) + time_offset for x in pred_times_f.readlines()]
            for t in times:
                full_times_f.write(f"{t}\n")
            time_offset = times[-1]

            pred_poses_f.close()
            pred_times_f.close()
            progress.update(1)

        progress.close()

        shutil.copy("config/calib.txt", f"{pred_full_dir}/calib.txt")

        full_poses_f.close()
        full_times_f.close()
        file_log.close()

