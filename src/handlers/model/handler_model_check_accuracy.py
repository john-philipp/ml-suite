import json
import logging
import os

import numpy as np

from src.parsers.enums import ArchitectureType, DataFormatType, ModelType
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler
from src.path_handler import PathHandler
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerModelCheckAccuracy(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.training_index = args.training_index
            self.dataset_index = args.dataset_index

            self.data_format = args.data_format
            self.model = args.model
            self.arch = args.arch

    def handle(self):
        args: HandlerModelCheckAccuracy.Args = self.args

        if args.arch == ArchitectureType.OPEN3D_ML:
            if args.data_format == DataFormatType.KITTI:
                if args.model == ModelType.RANDLANET:
                    log.info("Will check accuracy.")
                    self._accuracy_open3d_ml_kitti_randlanet(args.dataset_index, args.training_index)
                    return

        raise NotImplementedError()

    @staticmethod
    def _accuracy_open3d_ml_kitti_randlanet(dataset_index, training_index):

        dataset_path_handler = PathHandler("_generated/02-datasets")
        dataset_path = dataset_path_handler.get_path(dataset_index)

        training_path_handler = PathHandler("_generated/03-trainings")
        training_path = training_path_handler.get_path(training_index)

        test_seqs_dir = f"{training_path}/randlanet/test/sequences"
        test_seqs = os.listdir(test_seqs_dir)
        test_seqs.sort()

        label_dict = dict()
        global_equal = 0
        global_size = 0
        for test_seq in test_seqs:
            orig_seq_dir = f"{dataset_path}/dataset/sequences/{test_seq}"
            test_seq_dir = f"{test_seqs_dir}/{test_seq}"

            labels_dir_true = f"{orig_seq_dir}/labels"
            labels_dir_pred = f"{test_seq_dir}/predictions"

            labels_true = os.listdir(labels_dir_true)
            labels_pred = os.listdir(labels_dir_pred)

            labels_true.sort()
            labels_pred.sort()

            assert len(labels_true) == len(labels_pred)

            seq_equal = 0
            seq_size = 0
            for i, label_true in enumerate(labels_true):
                label_pred = labels_pred[i]

                arr_true = np.fromfile(f"{labels_dir_true}/{label_true}", dtype=np.uint32)
                arr_pred = np.fromfile(f"{labels_dir_pred}/{label_pred}", dtype=np.uint32)
                assert arr_true.size == arr_pred.size

                arr_equal = np.sum(arr_true == arr_pred)

                for j in range(0, len(arr_true)):
                    x = arr_true[j]
                    y = arr_pred[j]
                    y_int = int(y)
                    x_int = int(x)
                    if y_int not in label_dict:
                        label_dict[y_int] = {"total": 0, "errors": 0}
                    if x_int not in label_dict:
                        label_dict[x_int] = {"total": 0, "errors": 0}
                    label_dict[y_int]["total"] += 1
                    if x != y:
                        label_dict[int(y)]["errors"] += 1
                        label_dict[int(x)]["errors"] += 1

                print(f"{test_seq}.{label_true}: Accuracy {arr_equal / arr_true.size:3.03f}")

                seq_equal += arr_equal
                seq_size += arr_true.size

            print(f"{test_seq}: Accuracy {seq_equal / seq_size:3.03f}")
            global_equal += seq_equal
            global_size += seq_size

        print(f"Overall accuracy {global_equal / global_size:3.03f}")
        for x, y in label_dict.items():
            if y["total"] > 0:
                label_dict[x]["ratio"] = y["errors"] / y["total"]
        print(json.dumps(label_dict, indent=2))


