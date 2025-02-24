import os
from abc import ABC

from src.data_provider.error import EndOfSplitError
from src.path_helper import PathHelper, PathHelperConfig


class SequencePathProvider(ABC):
    def __init__(self, database_path, sequence_split):
        self.database_path = database_path
        self.ph = PathHelper(database_path, config=PathHelperConfig(auto_rollback=True))
        self.ph.sequences().commit()
        self.sequence_split = sequence_split
        self.sequence_data_count = None
        self.sequence_index = None
        self.sequence_path = None
        self.split_index = -1
        self.data_index = -1

        self.read_next_sequence()

    def get_next(self):
        self.data_index += 1
        if self.data_index >= self.sequence_data_count:
            self.data_index = 0
            self.read_next_sequence()
        return self.sequence_path, self.sequence_index, self.data_index

    def read_next_sequence(self):
        self.split_index += 1
        try:
            sequence_index = self.sequence_split[self.split_index]
            try:
                sequence_index = int(sequence_index)
            except ValueError:
                pass
            self.sequence_index = sequence_index
        except IndexError:
            raise EndOfSplitError()
        self.sequence_path = self.ph.index(self.sequence_index).path()
        self.sequence_data_count = self.max_data_in_sequence()

    def max_data_in_sequence(self):
        # This is more stateful than I'd really like. Inject?
        return len(os.listdir(self.ph.index(self.sequence_index).bins().path()))
