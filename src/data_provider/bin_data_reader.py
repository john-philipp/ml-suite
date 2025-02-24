import os

import numpy as np

from src.data_provider.error import EndOfSequenceError
from src.data_provider.interface import _IDataReader
from src.path_helper import PathHelper


class BinDataReader(_IDataReader):
    def __init__(self):
        self.sequence_path = None
        self.bin_dir = None
        self.bin_list = []
        self.total = None

    def get_progress(self):
        return self.total

    def update_sequence(self, sequence_path):
        if sequence_path is None or self.sequence_path != sequence_path:
            self.sequence_path = sequence_path
            self.bin_dir = PathHelper(sequence_path).bins().path()
            self.bin_list = os.listdir(self.bin_dir)
            self.bin_list.sort()
            self.total = len(self.bin_list)

    def get_data(self, sequence_path, data_index):
        self.update_sequence(sequence_path)
        try:
            bin_path = os.path.join(self.bin_dir, self.bin_list[data_index])
            bin_data = np.fromfile(bin_path, dtype=np.float32)
            # Note: This assumes 4D data. Might want to make this flexible?
            bin_data.resize(int(len(bin_data) / 4), 4)
            return bin_data
        except IndexError:
            raise EndOfSequenceError()
