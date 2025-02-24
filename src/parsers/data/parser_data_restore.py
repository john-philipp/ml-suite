from src.parsers.enums import DataActionType
from src.parsers.interfaces import _Parser


class ParserDataRestore(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Restore a data snapshot.",
            name=DataActionType.RESTORE, aliases=[DataActionType.LOAD],
            help="Restore a snapshot.")

        parser.add_argument(
            "--id",
            help=f"Alias to restore from (-1 == latest) (default=%(default)s).",
            default=-1)

        parser.add_argument(
            "--merge",
            help=f"Dirs to merge into existing (string delimited with ',').")

        return parser
