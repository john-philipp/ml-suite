from src.parsers.enums import DataActionType
from src.parsers.interfaces import _Parser


class ParserDataAnnotate(_Parser):
    @classmethod
    def add_args(cls, parent_parser):
        
        parser = parent_parser.add_parser(
            description="Annotate a snapshot.",
            name=DataActionType.ANNOTATE,
            help="Annotate a snapshot.")

        parser.add_argument(
            "--id",
            help="Annotate previously saved snapshot by id.")

        parser.add_argument(
            "--add-annotations", "--add",
            help="Add annotations {key}={value} kvps (no spaces).",
            nargs="+")

        parser.add_argument(
            "--rm-annotations", "--rm",
            help="Remove annotation for {key}s provided.",
            nargs="+")

        return parser
