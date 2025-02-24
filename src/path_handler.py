import os


def get_index_from_name(name):
    return int(name.split("_")[0])


class PathHandler:
    def __init__(self, path, splitter=None):
        self.path = path
        self.sub_paths = os.listdir(self.path)
        self.sub_paths.sort()
        self.index_to_path = {}
        self.rev_index_to_path = {}
        self.string_indexes = {}
        for sub_path in self.sub_paths:
            try:
                self.index_to_path[(splitter or get_index_from_name)(sub_path)] = sub_path
            except ValueError:
                self.string_indexes[sub_path] = sub_path

        # Allow reverse index.
        rev_index = -1
        while True:
            try:
                self.rev_index_to_path[rev_index] = self.sub_paths[rev_index]
                rev_index -= 1
            except IndexError:
                break

    def get_latest_path(self):
        try:
            return f"{self.path}/{self.sub_paths[-1]}"
        except IndexError:
            return ""

    def get_latest_index(self):
        try:
            return max(self.index_to_path.keys() or [0])
        except IndexError:
            return -1

    def get_next_index(self):
        return self.get_latest_index() + 1

    def has_index(self, index):
        return index in self.index_to_path or index in self.rev_index_to_path or index in self.string_indexes

    def get_sub_path(self, index):
        # assert index, f"Index must be != 0"
        if isinstance(index, str):
            return self.string_indexes[index]
        if index < 0:
            return self.rev_index_to_path[index]
        else:
            return self.index_to_path[index]

    def get_path(self, index, suffix="", ok_missing=False):
        if not self.has_index(index):
            if not ok_missing:
                raise AssertionError(f"Missing path in {self.path} for index: {index}")
            return f"{self.path}/{index:04}_{suffix}"
        return f"{self.path}/{self.get_sub_path(index)}"

    def get_next_path(self, suffix):
        return f"{self.path}/{self.get_next_index():04}_{suffix}"
