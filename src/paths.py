import os

from src.file_helpers import pj


class _Paths:

    def __init__(self):
        self.generated = "_generated"
        self.msg = pj(self.generated, "_msg").__str__()
        self.str = pj(self.generated, "_str").__str__()
        self.json = pj(self.generated, "_json").__str__()
        self.kitti = pj(self.generated, "kitti")
        self.training = pj(self.generated, "training")
        self.predictions = pj(self.generated, "predictions")
        self.snapshots = f"{os.environ['HOME']}/.mls/snapshots"
        self.results = f"{os.environ['HOME']}/.mls/results"
        # Keep "removed" data for sanity. Removed on restart.
        self.save_sanity = "/tmp/.mls/sanity"

    def all_dirs(self):
        return self.__dict__.values()


PATHS = _Paths()
