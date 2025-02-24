from src.data_provider.interface import _ILineDataReader
from src.path_helper import PathHelper


class PoseDataReader(_ILineDataReader):
    def update_sequence(self, sequence_path):
        if sequence_path is None or self.sequence_path != sequence_path:
            self.sequence_path = sequence_path
            self.file_path = PathHelper(sequence_path).poses().path()
            self.read_file(self.file_path)

    def get_data(self, sequence_path, data_index):
        data_line = super().get_data(sequence_path, data_index)
        return [float(x) for x in data_line.split(' ')[3::4]]
