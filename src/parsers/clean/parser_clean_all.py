from src.parsers.enums import CleanActionType
from src.parsers.interfaces import _Parser


class ParserCleanAll(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Clean everything (excluding snapshots and results).",
            name=CleanActionType.ALL,
            help="Clean all data (excluding snapshots and results).")

        return parser
