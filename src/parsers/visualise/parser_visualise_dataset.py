from src.parsers.enums import VisualiseActionType
from src.parsers.interfaces import _Parser


class ParserVisualiseDataset(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Visualise a dataset in point_labeler.",
            name=VisualiseActionType.DATASET,
            help="Visualise a dataset.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Use this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        # Prefix values with ":" to not misinterpret leading "-" as new flag.
        # Default defaults to from-file, if missing uses value.
        parser.add_argument(
            "--screenshot-config", "-s",
            help="Use this screenshot config [:zoom:up:right|from-file] (default=%(default)s).",
            default="from-file|:-30:0:-10")

        parser.add_argument(
            "--take-screenshot", "-t",
            help="Take screenshot.",
            action="store_true")

        parser.add_argument(
            "--write-screenshot-config", "-w",
            help="Write screenshot config to file.",
            action="store_true")

        return parser
