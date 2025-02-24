from abc import ABC, abstractmethod

from src.data_provider.error import EndOfSequenceError
from src.data_provider.sequence_data_provider import SequencePathProvider


class _IDataReader(ABC):
    @abstractmethod
    def get_data(self, sequence_path, data_index):
        raise NotImplementedError()

    @abstractmethod
    def update_sequence(self, sequence_path):
        raise NotImplementedError()


class _ILineDataReader(_IDataReader, ABC):
    def __init__(self):
        self.sequence_path = None
        self.file_data = None
        self.file_path = None
        self.total = None

    def read_file(self, file_path):
        with open(file_path) as f:
            self.file_data = [x.strip() for x in f.readlines()]
            self.total = len(self.file_data)

    def get_data(self, sequence_path, data_index):
        self.update_sequence(sequence_path)
        try:
            return self.file_data[data_index]
        except IndexError:
            raise EndOfSequenceError()


class _IMsgConverter(ABC):
    @abstractmethod
    def convert(self, data, header):
        raise NotImplementedError()


class _IDataProvider(ABC):
    def __init__(self, database_path, sequence_split, data_reader: _IDataReader):
        self.database_path = database_path
        self.sequence_split = sequence_split
        self.sequence_path_provider = SequencePathProvider(database_path, sequence_split)
        self.data_reader = data_reader

    def get_next(self):
        sequence_path, sequence_index, data_index = self.sequence_path_provider.get_next()
        return self.data_reader.get_data(sequence_path, data_index), sequence_index, data_index
