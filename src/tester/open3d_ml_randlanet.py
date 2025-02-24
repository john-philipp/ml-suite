import os.path
import shutil
import threading
import time
import warnings
import logging

import humanize
import numpy as np
from tqdm import tqdm

from src.path_helper import PathHelper
from src.jinja_yaml_loader import JinjaYamlLoader
from src.file_logger import FileLogger
from src.methods import round_float, set_data, bindings_from_args, get_file_md5, get_data, read_yaml
from src.path_handler import PathHandler
from src.suppress_std_err import SuppressStdErr
from src.tester.interfaces import ITester
from src.this_env import GLOBALS

# Ignore all warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


log = GLOBALS.log


class GpuMemoryGetter(threading.Thread):
    def __init__(self, sleep_s=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_s = sleep_s
        self.stopping = False
        self.total_used_max = 0
        self.total_available = 0
        self.devices = []

    def run(self):

        try:
            import nvidia_smi
        except ImportError:
            # Package is missing. No need to continue.
            log.warning("Missing package 'nvidia_smi', can't track GPU resources.")
            return

        nvidia_smi.nvmlInit()

        my_pid = os.getpid()
        total_memory = 0
        total_used_max = 0

        device_count = nvidia_smi.nvmlDeviceGetCount()

        # Get totals.
        for i in range(0, device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            name = nvidia_smi.nvmlDeviceGetName(handle).decode()
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            self.devices.append(name)
            total_memory += info.total

        self.total_available = total_memory / 1024 / 1024

        while not self.stopping:
            curr_total_used = 0
            for i in range(0, device_count):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                processes = nvidia_smi.nvmlDeviceGetComputeRunningProcesses(handle)
                for process in processes:
                    if not process.pid == my_pid:
                        continue
                    curr_total_used += process.usedGpuMemory

            total_used_max = max(total_used_max, curr_total_used)
            time.sleep(self.sleep_s)

        self.total_used_max = total_used_max / 1024 / 1024
        nvidia_smi.nvmlShutdown()

    def join(self, *args, **kwargs):
        self.stopping = True
        super().join(*args, **kwargs)


class _TesterOpen3dMLRandlanet(ITester):

    def __init__(self, args, config_name="config.yml"):
        self._config_name = config_name
        self._args = args

    def test(self):

        memory_getter = GpuMemoryGetter()
        memory_getter.start()

        args = self._args
        bindings = bindings_from_args(args.bindings_kvp, args.bindings_json)

        dataset_path_handler = PathHandler("_generated/02-datasets")
        dataset_path = dataset_path_handler.get_path(args.dataset_index)

        training_path_handler = PathHandler("_generated/03-trainings")
        training_path = training_path_handler.get_path(args.training_index)

        # How many epochs trained into checkpoint?
        epochs_trained = None
        try:
            training_info_path = PathHelper(training_path).info().path()
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
                    training_action, "summary.input.epochs_trained", default=0)

        # Remove any previous test results (will fail o/w).
        shutil.rmtree(f"{training_path}/randlanet/logs/cache", ignore_errors=True)
        shutil.rmtree(f"{training_path}/randlanet/test", ignore_errors=True)

        file_log = FileLogger(training_path)
        file_log.log(f"Handler: {self.__class__.__name__}")
        file_log.log(f"Bindings: {bindings}")
        file_log.log(f"Dataset: {file_log.link(dataset_path)}")

        # TODO this might not always be correct.
        #  I'm wondering if there'll be scenarios in which it's valid to specify
        #  bindings during testing. That are allowed to conflict with testing
        #  bindings? Right now, we just blanket error. Effectively disabling
        #  it, while keeping the infrastructure alive, in case we want it later.
        #  Possible ideas include to add an override bindings flag? But it
        #  depends on the problem.
        bindings_path = f"{training_path}/bindings.yml"
        if not os.path.exists(bindings_path):
            file_log.log("Missing bindings.yml in training.")
            file_log.close()
            raise AssertionError()

        previous_bindings = read_yaml(bindings_path)
        if bindings:
            file_log.log(f"I'm ignoring specified bindings: {bindings}")
        file_log.log(f"Defaulting to pre-existing training bindings: {previous_bindings}")
        bindings = previous_bindings

        config_path = f"{dataset_path}/{self._config_name}"
        if not config_path:
            raise AssertionError("No config path set.")
        if not os.path.isfile(config_path):
            raise AssertionError(f"Config path doesn't exist: {config_path}")

        class_weights = []
        # TODO make part of path_helper?
        class_weights_path = f"{training_path}/weights.yml"
        if not os.path.exists(class_weights_path):
            file_log.log(f"Using default class weights as per semantic kitti open3d-ml.")
        else:
            class_weight_data = read_yaml(class_weights_path)
            class_weights = class_weight_data["weights"]
            file_log.log(f"Using weights:     {class_weights}")
            file_log.log(f"Read from here:    {file_log.link(class_weights_path)}")
            file_log.log(f"Based on datasets: {class_weight_data['datasets']}")

        file_log.keep_file(config_path)
        file_log.log("Handling imports...")

        import torch
        import open3d.ml.torch as ml3d
        from open3d._ml3d.utils import Config
        from open3d._ml3d.torch import RandLANet, SemanticSegmentation
        from open3d._ml3d.torch import SemSegMetric, filter_valid_label

        # Not necessary. Left for reference.
        torch.cuda.memory.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()

        config_loader = JinjaYamlLoader(config_path, lambda **x: Config(x))
        config = config_loader.load(dataset_path=dataset_path, training_path=training_path, **bindings)

        if not config.dataset["test_split"]:
            file_log.log("No test split found.")
            file_log.close()
            raise AssertionError("No test split found.")

        split = f"{args.split}_split"
        file_log.log(f"Using split: {split}")
        config.dataset["test_split"] = config.dataset[split]
        if args.tests_num > 0:
            config.dataset["test_split"] = config.dataset["test_split"][:args.tests_num]

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
        config.model["num_points"] = int(num_points)
        config.model["num_layers"] = int(num_layers)
        config.model["num_neighbors"] = int(num_neighbors)

        dataset = ml3d.datasets.SemanticKITTI(**config.dataset)
        chosen_split = dataset.get_split("test")
        model = RandLANet(**config.model)

        pipeline = SemanticSegmentation(model=model, dataset=dataset, **config.pipeline)

        if not args.checkpoint_path:
            checkpoints_path = f"{training_path}/randlanet/logs/RandLANet_SemanticKITTI_torch/checkpoint"
            checkpoints = os.listdir(checkpoints_path)
            checkpoints.sort()
            if not checkpoints:
                file_log.log(f"No checkpoints in: {file_log.link(checkpoints_path)}")
                file_log.close()
                raise AssertionError(f"No checkpoints in: {checkpoints_path}")
            chosen_checkpoint_path = f"{checkpoints_path}/{checkpoints[-1]}"
        else:
            if not os.path.exists(args.checkpoint_path):
                raise AssertionError(f"Checkpoint specified not found: {args.checkpoint_path}")
            chosen_checkpoint_path = args.checkpoint_path

        chosen_checkpoint_size = round_float(os.path.getsize(chosen_checkpoint_path) / 1024 / 1024, 2)
        chosen_checkpoint_md5 = get_file_md5(chosen_checkpoint_path)[:8]

        pipeline.load_ckpt(chosen_checkpoint_path)

        file_log.log(f"Loaded checkpoint: {chosen_checkpoint_path} sha={chosen_checkpoint_md5}")
        file_log.add_infos(local=True, checkpoint=chosen_checkpoint_path)

        performed_action = dict(
            timestamp=file_log.timestamp,
            type="test",
            training=file_log.link(training_path),
            dataset=file_log.link(dataset_path),
            dataset_size=self._get_dir_size(dataset_path),
            checkpoint_path=file_log.link(chosen_checkpoint_path),
            checkpoint_size=chosen_checkpoint_size,
            checkpoint_md5=chosen_checkpoint_md5,
            summary=dict(
                input=dict(
                    epochs_trained=epochs_trained,
                    len_split=len(chosen_split),
                    neighbours=config.model["num_neighbors"],
                    points=config.model["num_points"],

                ),
                output=dict(
                )
            ),
            logs=file_log.link(file_log.get_log_dir()),
            args=args.__dict__)

        # Inference results here are printable.
        start_time = time.time()
        if args.inference_only:
            performed_action["type"] = "inference"
            file_log.log("Beginning inference...")
            len_split = len(chosen_split)
            i = 0

            progress = tqdm(total=len_split)
            inference_accs = []
            inference_ious = []
            inf_times = []

            while i < len_split:

                data = chosen_split.get_data(i)
                metric = SemSegMetric()

                with SuppressStdErr():

                    start_time_inf = time.time()
                    result = pipeline.run_inference(data)
                    end_time_inf = time.time()

                    inf_times.append(end_time_inf - start_time_inf)

                    valid_scores, valid_labels = filter_valid_label(
                        torch.tensor(result['predict_scores']),
                        torch.tensor(data['label']), model.cfg.num_classes,
                        model.cfg.ignored_label_inds, "cuda")

                    metric.update(valid_scores, valid_labels)

                progress.update(1)
                acc = metric.acc()
                if acc:
                    inference_accs.append(acc[-1])
                iou = metric.iou()
                if iou:
                    inference_ious.append(iou[-1])
                i += 1

            set_data(
                performed_action,
                round_float(np.mean(inference_accs), 3),
                "summary.output.acc")
            set_data(
                performed_action,
                round_float(np.mean(inference_ious), 3),
                "summary.output.iou")
            set_data(
                performed_action,
                round_float(np.mean(inf_times), 3),
                "summary.output.mean_inf_time")
            set_data(
                performed_action,
                round_float(sum(inf_times) / len_split, 3),
                "summary.output.avg_inf_time")

            file_log.log("Finished inference.")
        else:
            performed_action["type"] = "test"
            file_log.log("Beginning testing...")
            try:
                pipeline.run_test()
            except TypeError:
                # Can happen if no labels, yet.
                pass
            else:
                set_data(
                    performed_action,
                    round_float(pipeline.metric_test.acc()[-1], 3),
                    "summary.output.acc")
                set_data(
                    performed_action,
                    round_float(pipeline.metric_test.iou()[-1], 3),
                    "summary.output.iou")

            file_log.log("Finished testing.")

        end_time = time.time()

        memory_getter.join()
        set_data(performed_action, memory_getter.total_available, "memory.total_available_mb")
        set_data(performed_action, memory_getter.total_used_max, "memory.total_max_used_mb")
        set_data(performed_action, memory_getter.devices, "memory.devices")

        performed_action["execution_time_s"] = round_float(end_time - start_time, 3)
        file_log.add_infos(root=True, append=True, performed_actions=performed_action)
        file_log.add_infos(local=True, performed_actions=[performed_action])
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
