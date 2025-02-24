from src.parsers.enums import DataActionType
from src.parsers.interfaces import _Parser


class ParserDataSave(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Save current state of _generated/ dir as snapshot.",
            name=DataActionType.SAVE, aliases=[DataActionType.STORE],
            help="Save current state of _generated/ dir.")

        parser.add_argument(
            "--alias", "-l",
            help="Specify alias (default=%(default)s).",
            default="latest")

        parser.add_argument(
            "--into-id", "-i",
            help="Store into existing snapshot (faster).")

        parser.add_argument(
            "--annotations", "-a",
            help="Annotations {key}={value} kvps (no spaces).",
            nargs="+")

        return parser
