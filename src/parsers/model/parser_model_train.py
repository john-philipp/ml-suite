from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelTrain(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Train a model on some labelled data.",
            name=ModelActionType.TRAIN,
            help="Train based on dataset.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Based on this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--training-index", "-t",
            help="Based on this training (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--weights-index", "-w",
            help="Use these weights (-1 == defaults) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--train-into", "-i",
            help="Train into training-index (default=%(default)s).",
            action="store_true")

        parser.add_argument(
            "--epochs", "-e",
            help="Number of epochs (default=%(default)s)",
            default=10,
            type=int)

        parser.add_argument(
            "--split", "-s",
            help="Use this split as per config (default=%(default)s).",
            choices=["full", "all", "training", "validation", "test", "sparse"],
            default="sparse")

        parser.add_argument(
            "--bindings-kvp", "-b",
            help="Define bindings as kvp pairs (key:value[:int|float|str|bool]). Takes priority over JSON.",
            nargs="+")

        parser.add_argument(
            "--bindings-json", "-b2",
            help="Define bindings as a JSON string for more complex bindings.")

        return parser
