import os

import numpy as np
from tqdm import tqdm

from src.parsers.enums import DataFormatType
from src.parsers.interfaces import _Args
from src.file_helpers import rm
from src.handlers.interfaces import _Handler
from src.methods import read_label_data
from src.path_handler import PathHandler
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerModelMapLabels(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.dataset_index = args.dataset_index
            self.data_format = args.data_format
            self.mappings = args.mappings

    def handle(self):
        args: HandlerModelMapLabels.Args = self.args

        if args.data_format == DataFormatType.KITTI:
            self.map_labels_kitti("_generated/02-datasets", args.dataset_index, args.mappings)
            return

        raise NotImplementedError()

    @classmethod
    def map_labels_kitti(cls, path, dataset_index, mappings_s):
        dataset_path_handler = PathHandler(path)
        dataset_path = dataset_path_handler.get_path(dataset_index)
        log.info(f"Dataset:   {dataset_path}")
        seqs_path = f"{dataset_path}/dataset/sequences"
        seq_dirs = os.listdir(seqs_path)
        seq_dirs.remove(".full")
        seq_dirs.sort()

        # Clean out existing sequences.
        # Need to prepare again.
        for seq_dir in seq_dirs:
            rm(f"{seqs_path}/{seq_dir}")

        # We map in order.
        mappings = []
        for mapping in mappings_s:
            _, map_from, map_to = mapping.split(":")
            mappings.append((int(map_from), int(map_to)))
        log.info(f"Mappings: {mappings}")

        labels_path = f"{seqs_path}/.full/labels"
        label_files = os.listdir(labels_path)
        label_files.sort()

        progress = tqdm(desc="Label files", total=len(label_files))
        for label_file in label_files:
            label_file_path = f"{labels_path}/{label_file}"
            label_data = read_label_data(label_file_path)
            for map_from, map_to in mappings:
                if map_from == -1:
                    label_data[:] = map_to
                    continue
                label_data[label_data == int(map_from)] = int(map_to)
            with open(label_file_path, "wb") as f:
                label_data.tofile(f)
            progress.update(1)
        progress.close()

