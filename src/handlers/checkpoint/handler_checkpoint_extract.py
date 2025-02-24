import json
import os.path
import re
import shutil

from src.handlers.results import HandlerResultsCollect
from src.file_logger import FileLogger
from src.file_helpers import mk, mv
from src.jinja_yaml_loader import JinjaYamlLoader
from src.methods import read_yaml, write_yaml, get_git_sha, get_file_md5, get_timestamp, find_marks
from src.path_helper import PathHelper
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler


class HandlerCheckpointExtract(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.training_index = args.training_index
            self.version = args.version
            self.type = args.type

    def handle(self):
        args: HandlerCheckpointExtract.Args = self.args

        file_log = FileLogger()
        version_pattern = r"^v[0-9]+(\.[0-9]+)?(\.[0-9]+)?$"
        if not args.version.startswith("v"):
            args.version = "v" + args.version

        match = re.match(version_pattern, args.version)
        if not match:
            file_log.log_and_raise(f"Invalid version format (need {version_pattern}): {args.version}", AssertionError)

        name = f"{args.type}-{args.version}"

        if not os.path.exists(".ckpt-data"):
            file_log.log_and_raise("Missing local symlink or directory '.ckpt-data'.", AssertionError)

        # Generate new dir in .ckpt-data.
        ckpt_data_dir = PathHelper(".ckpt-data").next(suffix=name).path()
        if os.path.exists(ckpt_data_dir):
            file_log.log_and_raise(f"Checkpoint data dir '{ckpt_data_dir}' already exists.", AssertionError)

        file_log.log(f"Will write here: {ckpt_data_dir}")
        mk(ckpt_data_dir)

        # Get training data.
        training_path = PathHelper().generated().training(args.training_index).path()
        checkpoint_path = PathHelper(training_path).checkpoints().latest().path()
        file_log.log(f"Found checkpoint: {checkpoint_path}")
        info_path = f"{training_path}/info.yml"
        log_path = f"{training_path}/log.txt"

        shutil.copyfile(checkpoint_path, f"{ckpt_data_dir}/checkpoint.pth")
        shutil.copyfile(info_path, f"{ckpt_data_dir}/info.yml")
        shutil.copyfile(log_path, f"{ckpt_data_dir}/log.txt")

        # We get latest config.yml and bindings.yml from latest training log.
        # Find latest containing config.yml and bindings.yml.
        training_log_path = PathHelper(training_path).push("log").path()
        training_logs = os.listdir(training_log_path)
        training_logs.sort(reverse=True)

        latest_relevant_log = None
        for log_name in training_logs:
            def exist(dir_, *names):
                for name_ in names:
                    path = f"{dir_}/{name_}"
                    if not os.path.exists(path) or not os.path.isfile(path):
                        return False
                return True
            log_path = f"{training_log_path}/{log_name}"
            if exist(log_path, "config.yml", "bindings.yml"):
                latest_relevant_log = log_path
                break

        if not latest_relevant_log:
            file_log.log_and_raise(
                f"Couldn't find latest relevant log in training: {training_path}", AssertionError)

        bindings_path = f"{latest_relevant_log}/bindings.yml"
        config_path = f"{latest_relevant_log}/config.yml"

        file_log.log(f"Resolving config: {config_path}")
        file_log.log(f"Using bindings:   {bindings_path}")
        config = load_config(bindings_path, config_path, training_path="__ckpt_path__")
        write_yaml(f"{ckpt_data_dir}/config.yml", config)

        training_name = os.path.basename(training_path)
        file_log.log("Associating bags and datasets...")
        bag_by_dataset = HandlerResultsCollect.get_bag_by_dataset(
            training_path, ckpt_data_dir, file_log,
            copy_files=False, training_info_path=info_path, show_progress=True)

        file_log.log("Converting to csv...")
        info_data = HandlerResultsCollect.convert_info_to_csv(
            ckpt_data_dir, training_name, bag_by_dataset, file_log,
            include_all=False, training_info_path=info_path, show_progress=True)
        mv(f"{ckpt_data_dir}/results.csv", f"{ckpt_data_dir}/training_log.csv")

        first_training_action = next((
            action for action in info_data["performed_actions"] if action["type"] == "training"), None)
        if not first_training_action:
            file_log.log_and_raise(f"Couldn't find a training action in info data: {info_path}", AssertionError)

        checkpoint_md5, checkpoint_original_path = get_file_md5(checkpoint_path).split("  ")

        snapshot_uid = None
        snapshot_mark = find_marks(".", "uid.snapshot")
        if snapshot_mark:
            snapshot_uid = snapshot_mark[0]

        script = None
        script_mark = find_marks(".", "uid.script")
        if script_mark:
            script = script_mark[0]

        readme_data = dict(
            version=args.version,
            config_path=config_path,
            bindings_path=bindings_path,
            checkpoint_md5=checkpoint_md5,
            checkpoint_original_path=checkpoint_original_path,
            git_commit_sha=get_git_sha(),
            trained_on_data=bag_by_dataset,
            timestamp_first_trained=first_training_action["timestamp"],
            timestamp_extracted=get_timestamp(),
            based_on_snapshot=snapshot_uid,
            training_script=script
        )

        write_yaml(f"{ckpt_data_dir}/readme.txt", readme_data)
        file_log.log(f"Written here: {ckpt_data_dir}")
        file_log.log(json.dumps(readme_data, indent=2))
        file_log.close()


def load_config(bindings_path, config_path, **override_bindings):

    # Assuming non-nested bindings.
    bindings = read_yaml(bindings_path)
    bindings.update(override_bindings)

    config_loader = JinjaYamlLoader(config_path, dict)
    config = config_loader.load(**bindings)
    num_points = int(config["model"]["num_points"])
    num_layers = int(config["model"]["num_layers"])
    num_neighbors = int(config["model"]["num_neighbors"])

    # For lower points we need to reduce layers.
    if num_points < 512:
        num_layers = min(num_layers, 3)
    if num_points < 128:
        num_layers = min(num_layers, 2)
    if num_points < 32:
        num_layers = min(num_layers, 1)
    if num_points < 8:
        num_layers = min(num_layers, 0)
    if num_points < 2:
        raise ValueError("This won't work. Use points >= 2, please.")

    # Type preservation via jinja2 doesn't quite work here.
    config["model"]["num_points"] = int(num_points)
    config["model"]["num_layers"] = int(num_layers)
    config["model"]["num_neighbors"] = int(num_neighbors)
    return config
