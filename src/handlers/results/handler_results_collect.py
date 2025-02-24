import os

from tqdm import tqdm

from src.file_helpers import mk, cp
from src.jinja_yaml_loader import JinjaYamlLoader
from src.methods import get_timestamp, get_data, human_readable_to_bytes, round_float, get_git_sha
from src.parsers.interfaces import _Args
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler
from src.path_helper import PathHelper, PathHelperConfig


class HandlerResultsCollect(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.training_range = args.training_range

    def handle(self):
        args: HandlerResultsCollect.Args = self.args
        file_log = FileLogger()

        file_log.log(f"Handler: {self.__class__.__name__}")

        ph_config = PathHelperConfig(auto_rollback=True)

        def ph_factory():
            return PathHelper(config=ph_config)

        timestamp = get_timestamp()
        ph_results = ph_factory().results().push(timestamp).commit()

        training_index_start, training_index_end = args.training_range.split(":")
        if training_index_start == "":
            training_index_start = 1
        else:
            training_index_start = int(training_index_start)
        if training_index_end == "":
            training_index_end = -1
        else:
            training_index_end = int(training_index_end)

        # We allow for missing indexes.
        available_trainings = os.listdir(PathHelper().generated().trainings().path())
        available_trainings.sort()

        if training_index_end == -1:
            training_index_end = int(available_trainings[-1].split("_")[0])

        for i, available_training in enumerate(available_trainings):
            index = int(available_training.split("_")[0])
            if index == training_index_start:
                training_index_start = i
            if index == training_index_end:
                training_index_end = i

        trainings = available_trainings[training_index_start:training_index_end + 1]

        progress = tqdm(desc="Progress", total=len(trainings))
        for training_name in trainings:
            training_path = PathHelper().generated().trainings().push(training_name).path()
            results_path = ph_results.path()

            # Generate new results dir.
            mk(results_path)

            # cptree training w/o randlanet.
            training_name = os.path.basename(training_path)
            dst_dir = f"{results_path}/03-trainings/{training_name}"
            mk(dst_dir)

            cp(
                training_path, dst_dir, log_progress=False, rsync=True,
                rsync_args="--exclude randlanet --exclude log/**/ckpt_00000.pth")

            bag_by_dataset = self.get_bag_by_dataset(training_path, results_path, file_log)
            self.convert_info_to_csv(results_path, training_name, bag_by_dataset, file_log)
            progress.update(1)

        progress.close()

        file_log.log(f"Written to: {file_log.link(ph_results.path())}")
        file_log.close()

    @staticmethod
    def get_bag_by_dataset(
            training_path, results_path, file_log, copy_files=True, training_info_path=None, show_progress=False):

        if not training_info_path:
            # step through training/info.yml
            try:
                training_info_path = PathHelper(training_path).info(assert_exists=True).path()
            except AssertionError:
                file_log.log(f"Missing training info path, skipping: {training_path}")
                return {}

        training_info = JinjaYamlLoader(training_info_path, dict).load()
        actions = training_info["performed_actions"]
        bag_by_dataset = {}

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(desc="Progress", total=len(actions))

        for action in actions:
            dataset_path = action["dataset"].replace("file://", "")

            if progress_bar:
                progress_bar.update(1)

            if not os.path.exists(dataset_path):
                file_log.log(f"Missing dataset: {dataset_path}")
                continue

            dataset_name = os.path.basename(dataset_path)
            dst_dir = f"{results_path}/02-datasets/{dataset_name}"

            if copy_files:
                # Only do once per results set.
                if os.path.exists(dst_dir):
                    continue

                mk(dst_dir)
                cp(dataset_path, dst_dir, log_progress=False, rsync=True, rsync_args="--exclude dataset")

            try:
                dataset_info_path = PathHelper(dataset_path).info().path()
            except AssertionError:
                file_log.log(f"Couldn't find dataset info in: {dataset_path}")
            else:
                dataset_info = JinjaYamlLoader(dataset_info_path, dict).load()
                dataset_actions = dataset_info["performed_actions"]
                for dataset_action in dataset_actions:
                    if dataset_action["type"] == "convert":
                        recording_path = dataset_action["recording"].replace("file://", "")
                        recording_name = os.path.basename(recording_path)

                        if copy_files:
                            dst_dir = f"{results_path}/01-recordings/{recording_name}"
                            mk(dst_dir)
                            cp(recording_path, dst_dir, log_progress=False, rsync=True, rsync_args="--exclude msg")

                        recording_info_path = PathHelper(recording_path).info().path()
                        recording_info = JinjaYamlLoader(recording_info_path, dict).load()
                        bag_by_dataset[dataset_name] = recording_info["bagfile"]
                        break

        if progress_bar:
            progress_bar.close()

        return bag_by_dataset

    @staticmethod
    def convert_info_to_csv(
            results_dir, training_name, bag_by_dataset, file_log,
            include_all=True, training_info_path=None, show_progress=False):

        info_paths = []
        actions_ = []

        training_path = PathHelper(results_dir).trainings().push(training_name).path()
        if not training_info_path:
            training_info_path = PathHelper(training_path).info().path()

        try:
            info_yaml = JinjaYamlLoader(training_info_path, dict).load()
        except FileNotFoundError:
            file_log.log(f"Couldn't find info for training, skipping: {training_path}")
            return {}

        actions = info_yaml["performed_actions"]
        info_paths.append(training_info_path)
        actions_.append(actions)

        value_includes = [
            ("timestamp",               "timestamp"),
            ("git_sha",                 "git_sha"),
            ("type",                    "type"),
            ("arch",                    "args.arch"),
            ("model",                   "args.model"),
            ("training",                "training"),
            ("ckpt_sha",                "checkpoint_md5"),
            ("ckpt_size_mb",            "checkpoint_size"),
            ("dataset",                 "dataset"),
            ("dataset_size_mb",         "dataset_size"),

            ("recording",               "dataset_info.convert.recording"),
            ("bagfile",                 "dataset_info.bagfile"),
            ("epochs",                  "summary.input.epochs"),
            ("epochs_trained",          "summary.input.epochs_trained"),
            ("neighbours",              "summary.input.neighbours"),
            ("points",                  "summary.input.points"),

            ("train_pts",               "dataset_info.prepare.sequence_config.split_training.split_pts"),
            ("test_pts",                "dataset_info.prepare.sequence_config.split_test.split_pts"),
            ("seq_tps",                 "dataset_info.prepare.args.seq_tps"),
            ("seq_end",                 "dataset_info.prepare.args.seq_end"),

            ("labeled",                 ("dataset_info.label.labeled", "dataset_info.build-sample-data.labeled")),

            ("len_split",               "summary.input.len_split"),
            ("mean_loss",               "summary.output.loss"),
            ("mean_acc",                "summary.output.acc"),
            ("mean_iou",                "summary.output.iou"),
            ("mean_inf_time",           "summary.output.mean_inf_time"),
            ("time_s",                  "execution_time_s"),
            ("mem_used_mb",             "memory.total_max_used_mb"),
            ("mem_avail_mb",            "memory.total_available_mb"),
            ("devices",                 "memory.devices"),
            ("epsilon",                 "dataset_info.build-sample-data.epsilon"),
            ("label_counts",            (
                "dataset_info.label.label_counts", "dataset_info.build-sample-data.label_counts")),
        ]

        value_processors = {
            "dataset": lambda x: os.path.basename(x),
            "dataset_size_mb": lambda x: x if not x else round_float(human_readable_to_bytes(x) / 1024 / 1024, 2),
            "training": lambda x: os.path.basename(x),
            "recording": lambda x: os.path.basename(x),
            "devices": lambda x: ", ".join(x),
            "time_s": lambda x: x[0] if isinstance(x, list) else x,
            "git_sha": lambda x: get_git_sha(),
        }

        file_name_all = f"../results-all.csv"
        file_name_this = f"results.csv"

        file_names = []
        if include_all:
            file_names.append(file_name_all)
        file_names.append(file_name_this)

        progress_bar = None
        if show_progress:
            progress_bar = tqdm(desc="Progress", total=len(file_names) * len(actions) * len(actions_))

        for file_name in file_names:
            file_path = f"{results_dir}/{file_name}"
            file_exists = os.path.exists(file_path)

            existing_timestamps = set()
            if file_exists:
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        existing_timestamps.add(line.split(";")[0])

            with open(file_path, "a") as f:

                if not file_exists:
                    header_line = ""
                    for header, _ in value_includes:
                        if header_line:
                            header_line += ";"
                        header_line += f"{header}"

                    f.write(header_line + "\n")

                for actions in actions_:

                    for action in actions:

                        if progress_bar:
                            progress_bar.update(1)

                        dataset_path = action["dataset"].replace("file://", "")
                        dataset_name = os.path.basename(dataset_path)

                        action["dataset_info"] = dict(
                            build={},
                            label={},
                            prepare={},
                            convert={},
                            bagfile=bag_by_dataset[dataset_name] if dataset_name in bag_by_dataset else "",
                        )

                        try:
                            dataset_info_path = PathHelper().generated().datasets().push(dataset_name).info().path()
                        except AssertionError:
                            file_log.log(f"Missing info.yml in: {dataset_path}")
                        else:
                            dataset_info = JinjaYamlLoader(dataset_info_path, dict).load()
                            dataset_actions = dataset_info["performed_actions"]

                            for dataset_action in dataset_actions:
                                dataset_action_type = dataset_action["type"]
                                action["dataset_info"][dataset_action_type] = dataset_action

                        value_line = ""
                        for header, value_paths in value_includes:
                            if not isinstance(value_paths, tuple):
                                value_paths = (value_paths,)
                            value = ""
                            for value_path in value_paths:
                                # Use first value.
                                try:
                                    value = get_data(action, value_path, raise_on_missing=True)
                                    if value is None:
                                        value = ""
                                except KeyError:
                                    continue
                                break
                            if header in value_processors:
                                value = value_processors[header](value)
                            if header == "timestamp":
                                if value in existing_timestamps:
                                    # Skip existing.
                                    value_line = ""
                                    break
                            if value_line:
                                value_line += ";"
                            value_line += f"{str(value)}"

                        if value_line:
                            f.write(value_line + "\n")

        if progress_bar:
            progress_bar.close()

        return info_yaml
