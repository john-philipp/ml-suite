from src.parsers.enums import ModeType, MetaType
from src.parsers.interfaces import _Parser
from src.parsers.results.parser_results_collect import ParserResultsCollect


class ParserResults(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserResultsCollect
        ]

        parser = parent_parser.add_parser(
            description="Training result (metadata) handling.",
            name=ModeType.RESULTS,
            help="Handle results.")

        sub_parsers = parser.add_subparsers(
            dest=MetaType.ACTION,
            help="Action to take.")

        for sub_parser_cls in sub_parser_clss:
            sub_parser_cls.add_args(sub_parsers)

        return parser
