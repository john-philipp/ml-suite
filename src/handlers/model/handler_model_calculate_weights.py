import hashlib
import json
import os

from src.file_helpers import mk
from src.file_logger import FileLogger
from src.methods import read_yaml, write_yaml
from src.labels import LEARNING_MAP
from src.path_helper import PathHelper, PathHelperConfig
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerModelCalculateWeights(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.dataset_indexes = args.dataset_indexes

    def handle(self):
        args: HandlerModelCalculateWeights.Args = self.args

        file_log = FileLogger()

        ph_config = PathHelperConfig()
        ph_config.auto_rollback = True
        datasets_path = PathHelper(config=ph_config).generated().datasets().commit()
        available_datasets = datasets_path.available_indexes()

        selected = set()
        for index_range in args.dataset_indexes:
            range_select = self.read_dataset_index_range(available_datasets, index_range)
            selected.update(range_select)

        file_log.log(f"Selected datasets: {selected}")
        file_log.log("Looking for info.yml files...")

        # Find all info.ymls.
        info_ymls = []
        for index in available_datasets.keys():
            if index not in selected:
                continue
            info_yml = datasets_path.index(index).info().path()
            info_ymls.append(info_yml)
            if not os.path.exists(info_yml):
                file_log.log_and_raise(f"Missing file: {info_yml}")
            file_log.log(f"Found: {info_yml}")

        # Find latest label entry.
        last_label_action = None
        total_label_counts = {}

        file_log.log("Checking for label actions...")
        for info_yml in info_ymls:
            data = read_yaml(info_yml)
            actions = data["performed_actions"]
            for i in reversed(range(0, len(actions))):
                if actions[i]["type"] == "label":
                    last_label_action = actions[i]
                    break

            if not last_label_action:
                file_log.log_and_raise(
                    f"Info file {info_yml} doesn't reference a label action. You need to label all datasets first.")

            label_counts = last_label_action["label_counts"]
            for x, y in label_counts.items():
                total_label_counts[x] = total_label_counts.get(x, 0) + y

        # Weights range from 1:20. Shifted one to the left
        # (open3d-ml ignores 0:unlabeled in this list).
        # This is a little annoying. You just sort of have
        # to know, it seems. This is also true with predicted
        # labels coming back from pipeline.run_inference().
        file_log.log("Accumulating label counts...")
        class_weights = [0 for _ in range(0, 19)]

        # Get label counts.
        for label, count in total_label_counts.items():
            if label == 0:
                # Ignoring unlabeled. Decrement remainder by one.
                continue
            mapped_label = LEARNING_MAP[label] - 1
            class_weights[mapped_label] = count

        file_log.log(f"Class weights: {class_weights}")

        # Calculate hash for unique retrieval based on dataset selection.
        # We use full dataset names and sort for determinism.
        selected_list = list([y for x, y in available_datasets.items() if x in selected])
        selected_list.sort()
        json_string = json.dumps(selected_list)

        hash_object = hashlib.sha256(json_string.encode())
        sha_long = hash_object.hexdigest()

        # In principle. This isn't *that* unique. But should suffice.
        # If it's an issue, we can just extend from four characters.
        sha_short = sha_long[:4]

        weights_path = PathHelper(config=ph_config).generated().weights().commit()
        existing_weights = os.listdir(weights_path.path())
        existing_weights.sort()

        # Try to find existing weights with same sha.
        weight_path = None
        for existing in existing_weights:
            if existing.endswith(sha_short):
                file_log.log(f"Will update existing: {existing}")
                weight_path = weights_path.push(existing)

        if not weight_path:
            weight_path = weights_path.next(suffix=sha_short).path()
            file_log.log(f"Making new: {weight_path}")
            mk(weight_path)

        path = f"{weight_path}/weights_{sha_short}.yml"
        write_yaml(
            path,
            data=dict(
                sha=sha_long,
                datasets=selected_list,
                weights=class_weights))

        file_log.log(f"Written here: {file_log.link(path)}")
        file_log.close()

    @staticmethod
    def read_dataset_index_range(datasets, string):
        max_dataset = max(datasets)
        min_dataset = min(datasets)
        dataset_indexes = list(datasets.keys())
        dataset_indexes.sort()
        selected = set()

        parts = string.split(",")
        for part in parts:
            sub_parts = part.split(":")
            len_sub_parts = len(sub_parts)
            if len_sub_parts == 0:
                raise ValueError()
            elif len_sub_parts > 2:
                raise ValueError()
            elif len_sub_parts == 1:
                selected.add(int(sub_parts[0]))
            elif len_sub_parts == 2:
                start_index = int(sub_parts[0] or min_dataset)
                end_index = int(sub_parts[1] or max_dataset)
                for i in range(start_index, end_index + 1):
                    selected.add(i)

        for x in selected:
            if x < 0:
                selected.remove(x)
                selected.add(dataset_indexes[x])

        return selected

