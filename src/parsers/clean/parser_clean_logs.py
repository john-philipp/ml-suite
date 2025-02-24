from src.parsers.enums import CleanActionType
from src.parsers.interfaces import _Parser


class ParserCleanLogs(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Clean log(s).",
            name=CleanActionType.LOGS,
            help="Clean logs.")

        return parser
