import os.path
import shutil
import time
import uuid
import warnings
import logging

import humanize
import numpy as np
from ruamel.yaml import YAML

from src.jinja_yaml_loader import JinjaYamlLoader
from src.file_helpers import mk, cp
from src.file_logger import FileLogger
from src.methods import cp_markers, round_float, bindings_from_args, set_data, get_file_md5, get_data, read_yaml
from src.path_handler import PathHandler
from src.path_helper import PathHelper
from src.tester.open3d_ml_randlanet import GpuMemoryGetter
from src.this_env import GLOBALS
from src.trainer import ITrainer


# Ignore all warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


log = GLOBALS.log


class _TrainerOpen3dMlRandlanet(ITrainer):

    def __init__(self, args, config_name="config.yml"):
        self._config_name = config_name
        self._args = args

    def train(self):

        memory_getter = GpuMemoryGetter()

        if self._args.epochs > 0:
            memory_getter.start()

        args = self._args
        bindings = bindings_from_args(args.bindings_kvp, args.bindings_json)

        dataset_path_handler = PathHandler("_generated/02-datasets")
        dataset_path = dataset_path_handler.get_path(args.dataset_index)
        training_path_handler = PathHandler("_generated/03-trainings")

        # How many epochs overall in checkpoint.
        epochs_trained = 0

        if args.train_into and training_path_handler.has_index(args.training_index):
            training_path = training_path_handler.get_path(args.training_index)
            latest_checkpoint_path = PathHelper(training_path).checkpoints().latest().path()
            latest_checkpoint_md5 = get_file_md5(latest_checkpoint_path)[:8]
            log.info(f"Training into: {training_path} chkt_md5/sha={latest_checkpoint_md5}")
        else:
            training_uid = uuid.uuid4().hex
            training_path = training_path_handler.get_next_path(training_uid[:4])
            if args.training_index == -1:
                mk(training_path)
            else:
                if not training_path_handler.has_index(args.training_index):
                    raise AssertionError(f"No training found for index: {args.training_index}")
                base_on_path = training_path_handler.get_path(args.training_index)

                latest_checkpoint_path = PathHelper(base_on_path).checkpoints().latest().path()
                latest_checkpoint_md5 = get_file_md5(latest_checkpoint_path)[:8]

                log.info(f"Basing on existing training: {base_on_path} ckpt_md5/sha={latest_checkpoint_md5}")
                cp(base_on_path, training_path)

        cp_markers(dataset_path, training_path)
        try:
            training_info_path = PathHelper(training_path).info(assert_exists=True).path()
        except AssertionError:
            pass
        else:
            # Get latest. We could go backwards.
            training_info = JinjaYamlLoader(training_info_path, dict).load()
            training_action = None
            for action in training_info["performed_actions"]:
                if action["type"] == "training":
                    training_action = action
            if training_action:
                epochs_trained = get_data(
                    training_action, "summary.input.epochs_trained", default=epochs_trained)

        file_log = FileLogger(training_path)
        file_log.log(f"Handler: {self.__class__.__name__}")
        file_log.log(f"Bindings: {bindings}")
        file_log.log(f"Dataset: {file_log.link(dataset_path)}")

        bindings_path = f"{training_path}/bindings.yml"
        if os.path.exists(bindings_path):
            previous_bindings = read_yaml(bindings_path)

            # Default to previous bindings if exists and none specified.
            if not bindings:
                file_log.log(f"Defaulting to pre-existing bindings: {previous_bindings}")
                bindings = previous_bindings

            # Ensure consistent training.
            if previous_bindings != bindings:
                error_msg = "Trying to train with inconsistent bindings!"
                file_log.log(error_msg)
                file_log.log(f"Previous: {previous_bindings}")
                file_log.log(f"Current:  {bindings}")
                file_log.close()
                raise AssertionError(error_msg)
        else:
            # Write bindings to file.
            file_log.log(f"Writing bindings here: {bindings_path}")
            with open(bindings_path, "w") as f:
                YAML().dump(bindings, f)

        file_log.keep_file(bindings_path)

        config_path = f"{dataset_path}/{self._config_name}"
        if not config_path:
            raise AssertionError("No config path set.")
        if not os.path.isfile(config_path):
            raise AssertionError(f"Config path doesn't exist: {config_path}")

        # Try to find weights.
        class_weights = []  # Will use semantic kitti defaults in open3d-ml.
        if args.weights_index:
            class_weights_dir = PathHelper().generated().weights().index(args.weights_index).path()

            # Implicit assumption. First file. Might need extending later.
            class_weights_path = f"{class_weights_dir}/{os.listdir(class_weights_dir)[0]}"

            class_weight_data = read_yaml(class_weights_path)
            class_weights = class_weight_data["weights"]

            # TODO raise error when weights already exist and a different one
            #   has been specified on run?
            file_log.log(f"Using weights:     {class_weights}")
            file_log.log(f"Read from here:    {file_log.link(class_weights_path)}")
            file_log.log(f"Based on datasets: {class_weight_data['datasets']}")
            shutil.copyfile(class_weights_path, f"{training_path}/weights.yml")
        else:
            file_log.log(f"Using default class weights as per semantic kitti open3d-ml.")

        file_log.keep_file(config_path)
        file_log.log("Handling imports...")

        import torch
        import open3d.ml.torch as ml3d
        from open3d._ml3d.utils import Config
        from open3d._ml3d.torch import RandLANet, SemanticSegmentation

        # Not necessary. Left for reference.
        torch.cuda.memory.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()

        config_loader = JinjaYamlLoader(config_path, lambda **x: Config(x))
        config = config_loader.load(dataset_path=dataset_path, training_path=training_path, **bindings)

        if not config.dataset["training_split"]:
            file_log.log("No training split found. Did you prepare, yet?")
            file_log.close()
            raise AssertionError("No training split found.")

        num_points = int(config.model["num_points"])
        num_layers = int(config.model["num_layers"])
        num_neighbors = int(config.model["num_neighbors"])

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
        config.pipeline["max_epoch"] = args.epochs - 1
        config.model["num_points"] = int(num_points)
        config.model["num_layers"] = int(num_layers)
        config.model["num_neighbors"] = int(num_neighbors)

        dataset = ml3d.datasets.SemanticKITTI(**config.dataset)
        model = RandLANet(**config.model)
        pipeline = SemanticSegmentation(model=model, dataset=dataset, **config.pipeline)

        if args.epochs == 0:
            file_log.log(f"Called with zero epochs, just set up training. Quitting.")
            file_log.close()
            return

        if args.epochs > 1:
            file_log.log(f"Note, executing with > 1 epochs: {args.epochs}")

        file_log.log(f"Beginning training (epochs={args.epochs} points={num_points} "
                     f"neighbors={num_neighbors} layers={num_layers})...")
        start_time = time.time()

        try:
            pipeline.run_train()
        except KeyboardInterrupt:
            file_log.log(
                "Training was interrupted by user, data/result collection will be affected, check epochs actually ran!")

        end_time = time.time()

        acc = pipeline.metric_train.acc()
        iou = pipeline.metric_train.iou()

        try:
            losses = pipeline.losses
        except AttributeError:
            losses = None

        latest_checkpoint_path = PathHelper(training_path).checkpoints().latest().path()
        latest_checkpoint_size = round_float(os.path.getsize(latest_checkpoint_path) / 1024 / 1024, 2)
        latest_checkpoint_md5 = get_file_md5(latest_checkpoint_path)[:8]

        epochs = config.pipeline["max_epoch"] + 1
        performed_action = dict(
            timestamp=file_log.timestamp,
            type="training",
            training=file_log.link(training_path),
            dataset=file_log.link(dataset_path),
            dataset_size=self._get_dir_size(dataset_path),
            checkpoint_size=latest_checkpoint_size,
            checkpoint_md5=latest_checkpoint_md5,
            summary=dict(
                input=dict(
                    epochs=epochs,
                    epochs_trained=epochs_trained + epochs,
                    neighbours=config.model["num_neighbors"],
                    points=config.model["num_points"],
                    len_split=len(config.dataset["training_split"])
                ),
                output=dict(
                    acc=round_float(acc[-1] if acc else -1, 3),
                    iou=round_float(iou[-1] if iou else -1, 3),
                    loss=round_float(np.mean(losses) if losses else -1, 3)
                )
            ),
            logs=file_log.link(file_log.get_log_dir()),
            execution_time_s=round_float(end_time - start_time, 2),
            args=args.__dict__)

        memory_getter.join()
        set_data(performed_action, memory_getter.total_available, "memory.total_available_mb")
        set_data(performed_action, memory_getter.total_used_max, "memory.total_max_used_mb")
        set_data(performed_action, memory_getter.devices, "memory.devices")

        file_log.log("Finished training.")
        file_log.add_infos(root=True, append=True, performed_actions=performed_action)
        file_log.add_infos(local=True, performed_actions=[performed_action])

        checkpoint_path = PathHelper(training_path).checkpoints().latest().path()
        file_log.keep_file(checkpoint_path)
        file_log.close()

    # TODO refactor.
    @staticmethod
    def _get_dir_size(directory_path):
        total_size = 0
        for dir_path, dir_names, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dir_path, filename)
                # Skip if it is a broken symbolic link
                if not os.path.islink(file_path):
                    total_size += os.path.getsize(file_path)

        # Convert to human-readable format
        return humanize.naturalsize(total_size)
