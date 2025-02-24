from src.parsers.clean.parser_clean_all import ParserCleanAll
from src.parsers.clean.parser_clean_datasets import ParserCleanDatasets
from src.parsers.clean.parser_clean_logs import ParserCleanLogs
from src.parsers.clean.parser_clean_predictions import ParserCleanPredictions
from src.parsers.clean.parser_clean_recordings import ParserCleanRecordings
from src.parsers.clean.parser_clean_trainings import ParserCleanTrainings
from src.parsers.enums import ModeType
from src.parsers.interfaces import _Parser


class ParserClean(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserCleanAll,
            ParserCleanDatasets,
            ParserCleanPredictions,
            ParserCleanRecordings,
            ParserCleanTrainings,
            ParserCleanLogs
        ]

        parser = parent_parser.add_parser(
            description="Clean generated data (trainings/datasets/recordings/...).",
            name=ModeType.CLEAN,
            help="Handle clean.")

        sub_parsers = parser.add_subparsers(
            dest="action",
            help="Action to take.")

        for sub_parser_cls in sub_parser_clss:
            sub_parser_cls.add_args(sub_parsers)

        return parser
