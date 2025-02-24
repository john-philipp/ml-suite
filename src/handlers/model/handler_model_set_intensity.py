import os

import numpy as np
from tqdm import tqdm

from src.path_helper import PathHelperConfig, PathHelper
from src.parsers.enums import DataFormatType
from src.parsers.interfaces import _Args
from src.file_helpers import rm
from src.handlers.interfaces import _Handler
from src.methods import read_label_data
from src.path_handler import PathHandler
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerModelSetIntensity(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.dataset_index = args.dataset_index
            self.data_format = args.data_format
            self.mappings = args.mappings

    def handle(self):
        args: HandlerModelSetIntensity.Args = self.args

        if args.data_format == DataFormatType.KITTI:
            self.set_intensity_kitti("_generated/02-datasets", args.dataset_index, args.mappings)
            return

        raise NotImplementedError()

    @classmethod
    def set_intensity_kitti(cls, path, dataset_index, mappings_s):
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
            _, for_label, set_intensity = mapping.split(":")
            mappings.append((int(for_label), float(set_intensity)))
        log.info(f"Mappings: {mappings}")

        labels_path = f"{seqs_path}/.full/labels"
        label_files = os.listdir(labels_path)
        label_files.sort()

        ph_config = PathHelperConfig(auto_rollback=True)
        ph = PathHelper(config=ph_config).sequence2(dataset_index).commit()

        progress = tqdm(desc="Label files", total=len(label_files))
        for i, _ in enumerate(label_files):
            label_path = ph.labels().label(i).path()
            bin_path = ph.bins().bin(i).path()

            label_data = read_label_data(label_path)
            bin_data = np.fromfile(bin_path, dtype=np.float32)
            bin_data.resize(int(len(bin_data) / 4), 4)

            changed = False
            for for_label, set_intensity in mappings:
                # -1 corresponds to `all`.
                if for_label == -1:
                    bin_data[:, 3] = set_intensity
                    continue
                for_indexes = np.where(label_data == for_label)[0]
                if len(for_indexes) > 0:
                    bin_data[for_indexes, 3] = set_intensity
                    changed = True

            if changed:
                bin_data.tofile(bin_path)
            progress.update(1)
        progress.close()

