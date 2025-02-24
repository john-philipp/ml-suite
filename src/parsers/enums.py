from src.parsers.interfaces import _Enum


class MetaType(_Enum):
    MODE = "mode"
    ACTION = "action"


class ModeType(_Enum):
    VISUALISE = "visualise"
    RECORDING = "recording"
    MODEL = "model"
    DATA = "data"
    CLEAN = "clean"
    SCRIPTS = "scripts"
    MISC = "misc"
    RESULTS = "results"
    NODE = "node"
    CHECKPOINT = "checkpoint"


class NodeActionType(_Enum):
    PUB = "pub"
    INF = "inf"
    VIS = "vis"


class ActionType(_Enum):
    pass


class MiscActionType(_Enum):
    MAKE_DOCS = "make-docs"
    ECHO = "echo"


class DataActionType(ActionType):
    ANNOTATE = "annotate"
    RESTORE = "restore"
    CONVERT = "convert"
    CLEAN = "clean"
    SAVE = "save"
    LIST = "list"
    STORE = "store"
    LS = "ls"
    LOAD = "load"


class CleanActionType(ActionType):
    PREDICTIONS = "predictions"
    RECORDINGS = "recordings"
    TRAININGS = "trainings"
    DATASETS = "datasets"
    LOGS = "logs"
    ALL = "all"


class CheckpointActionType(ActionType):
    EXTRACT = "extract"


class RecordingActionType(ActionType):
    RECORD = "record"


class ScriptsActionType(ActionType):
    RUN = "run"


class ModelActionType(ActionType):
    CALCULATE_WEIGHTS = "calculate-weights"
    BUILD_SAMPLE_DATA = "build-sample-data"
    BUILD_PREDICTIONS = "build-predictions"
    BUILD_SEQUENCES = "build-sequences"
    CHECK_ACCURACY = "check-accuracy"
    FEATURE_STATS = "feature-stats"
    SET_INTENSITY = "set-intensity"
    MAP_LABELS = "map-labels"
    PREPARE = "prepare"
    LABEL = "label"
    TRAIN = "train"
    TEST = "test"


class ResultsActionType(ActionType):
    COLLECT = "collect"


class VisualiseActionType(ActionType):
    PREDICTIONS = "predictions"
    DATASET = "dataset"


class DataFormatType(_Enum):
    KITTI = "kitti"


class ArchitectureType(_Enum):
    OPEN3D_ML = "open3d-ml"


class ModelType(_Enum):
    RANDLANET = "randlanet"


class ModelPrepareSparseStrategy(_Enum):
    DISTANCE = "distance"
    SKIP = "skip"
