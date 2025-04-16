import os

from src.path_handler import PathHandler

_DELIMITER = "/"
_GENERATED = "_generated"
_LOGS = "00-logs"
_RECORDINGS = "01-recordings"
_DATASETS = "02-datasets"
_TRAININGS = "03-trainings"
_PREDICTIONS = "04-predictions"
_SCREENSHOTS = "05-screenshots"
_WEIGHTS = "06-weights"

# Note: This should live elsewhere. Can't import PATHS here. Duplicate definition.
_RESULTS = f"{os.environ['HOME']}/.mls/results"


class PathHelperConfig:
    def __init__(self, auto_rollback=False):
        self.auto_rollback = auto_rollback


ph_config = PathHelperConfig()


class PathHelper:
    def __init__(self, *paths, config=ph_config):
        self._config = config
        self._paths = ["."]
        self._paths2 = []
        if paths:
            self.push(*paths)
            self.commit()

    def cp(self):
        return PathHelper(self.path(skip_rollback=True), config=self._config)

    # TODO this currently overrides committed. Side-effect.
    def results(self):
        return self.push(_RESULTS)

    def screenshots(self):
        return self.push(_SCREENSHOTS)

    def weights(self):
        return self.push(_WEIGHTS)

    def rollback(self):
        self._paths2 = []
        return self

    def commit(self):
        if self._paths2:
            self._paths += self._paths2
            self._paths2 = []
        return self

    def push(self, *paths, assert_exists=False):
        for path in paths:
            if path[0] == "/":
                self._paths = ["/"]
            sub_paths = path.split(_DELIMITER)
            for sub_path in sub_paths:
                if sub_path:
                    self._paths2.append(sub_path)
        path = self.path(skip_rollback=True)
        if assert_exists:
            assert os.path.exists(os.path.abspath(path)), f"Missing path: {os.path.abspath(path)}"
        return self

    def pop(self, i=1):
        self._paths = self._paths[:-i]

    def path(self, rollback=False, skip_rollback=False):
        path = _DELIMITER.join(self._paths)
        if self._paths2:
            path = os.path.join(path, _DELIMITER.join(self._paths2))
        if not skip_rollback:
            if rollback or self._config.auto_rollback:
                self.rollback()
        return path

    def generated(self):
        path = self.path(skip_rollback=True)
        if os.path.isdir(os.path.join(path, _GENERATED)):
            self.push(_GENERATED)
        elif os.path.isdir(os.path.join(path, f"../{_GENERATED}")):
            self.push(f"../{_GENERATED}")
        path = self.path(skip_rollback=True)
        assert os.path.isdir(path), path
        return self

    def logs(self):
        self.push(_LOGS)
        return self

    def recordings(self):
        self.push(_RECORDINGS)
        return self

    def msg(self):
        self.push("msg")
        return self

    def datasets(self):
        self.push(_DATASETS)
        return self

    def available_indexes(self):
        current_path = self.path(rollback=True)
        items = os.listdir(current_path)
        items.sort()

        available_indexes = {}
        for item in items:
            x, y = item.split("_")
            available_indexes[int(x)] = item

        return available_indexes

    def sequences(self):
        self.push("dataset/sequences")
        return self

    def sequence(self, i=-1):
        if i == -1:
            self.push(".full")
        elif i == -2:
            self.push(".sparse")
        else:
            self.push(f"{i:02d}")
        return self

    def sequence2(self, database_index=-1, sequence_index=-1):
        return self.generated().datasets().index(database_index).sequences().sequence(sequence_index)

    def labels(self, **kwargs):
        self.push("labels", **kwargs)
        return self

    def label(self, i, **kwargs):
        self.push(f"{i:06d}.label", **kwargs)
        return self

    def bins(self, **kwargs):
        self.push("velodyne", **kwargs)
        return self

    def bin(self, i, **kwargs):
        self.push(f"{i:06d}.bin", **kwargs)
        return self

    def config(self):
        self.push("config.yml")
        return self

    def metadata(self):
        self.push("metadata.yml")
        return self

    def info(self, **kwargs):
        self.push("info.yml", **kwargs)
        return self

    def poses(self, **kwargs):
        self.push("poses.txt", **kwargs)
        return self

    def times(self, **kwargs):
        self.push("times.txt", **kwargs)
        return self

    def calib(self, **kwargs):
        self.push("calib.txt", **kwargs)
        return self

    def instances(self, **kwargs):
        self.push("instances.txt", **kwargs)
        return self

    def trainings(self):
        self.push(_TRAININGS)
        return self

    def training(self, index):
        self.trainings().index(index)
        return self

    def dataset(self, index):
        self.datasets().index(index)
        return self

    def checkpoints(self):
        self.push("randlanet/logs/RandLANet_SemanticKITTI_torch/checkpoint")
        return self

    def checkpoint(self, i=-1):

        def splitter(name):
            return int(name.split("_")[1].split(".")[0])

        path_handler = self._path_handler(
            self.path(skip_rollback=True), splitter=splitter)

        if i == -1:
            path = path_handler.get_latest_path()
        else:
            path = path_handler.get_path(i)

        self.push(os.path.basename(path))
        return self

    def predictions(self):
        self.push(_PREDICTIONS)
        return self

    def index(self, i):
        path_handler = self._path_handler(self.path(skip_rollback=True))
        self.push(path_handler.get_path(i).split(_DELIMITER)[-1])
        return self

    def latest(self):
        splitter = None
        path = self.path(skip_rollback=True)
        if path.endswith("checkpoint"):
            splitter = self.checkpoint_splitter
        path_handler = self._path_handler(path, splitter=splitter)
        self.push(path_handler.get_latest_path().split(_DELIMITER)[-1])
        return self

    def next(self, suffix=""):
        path_handler = self._path_handler(self.path(skip_rollback=True))
        self.push(path_handler.get_next_path(suffix=suffix).split(_DELIMITER)[-1])
        return self

    def save_sanity(self):
        return self.push("/tmp/.mls/sanity")

    @staticmethod
    def _path_handler(path, splitter=None):
        return PathHandler(path, splitter)

    @staticmethod
    def checkpoint_splitter(name):
        return int(name.split("_")[1].split(".")[0])

    def __repr__(self):
        return self.path(skip_rollback=True)

    def __str__(self):
        return self.path(skip_rollback=True)
