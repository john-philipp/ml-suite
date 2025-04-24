from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelLabel(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Label a dataset using the point_labeler.",
            name=ModelActionType.LABEL,
            help="Label a dataset.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Label this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--take-screenshot", "-t",
            help="Take screenshot.",
            action="store_true")

        parser.add_argument(
            "--kill-after", "-k",
            help="Kill labeler after seconds.",
            default=0,
            type=float)

        parser.add_argument(
            "--write-screenshot-config", "-w",
            help="Write screenshot config to file.",
            action="store_true")

        return parser
