from src.parsers.enums import ResultsActionType
from src.parsers.interfaces import _Parser


class ParserResultsCollect(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Training result (metadata) handling. Produces CSV for selected range of trainings.",
            name=ResultsActionType.COLLECT,
            help="Collect results.")

        parser.add_argument(
            "--training-range", "-t",
            help="Collect for these trainings ([{start:int}]:[{end:int}], ':' for all) (default=%(default)s).",
            default=":")

        return parser
