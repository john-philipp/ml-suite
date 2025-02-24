from src.parsers.enums import MiscActionType
from src.parsers.interfaces import _Parser


class ParserMiscEcho(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Echo messages to screen.",
            name=MiscActionType.ECHO,
            help="Echo to screen.")

        parser.add_argument(
            "--msgs", "-m",
            help="Echo these msgs.",
            required=True,
            nargs="+")

        return parser
