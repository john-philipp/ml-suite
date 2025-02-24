from src.parsers.interfaces import _Parser
from src.parsers.enums import DataActionType


class ParserDataList(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="List available data snapshots.",
            name=DataActionType.LIST, aliases=[DataActionType.LS],
            help="List available snapshots (recordings/datasets/trainings).")

        parser.add_argument(
            "--annotations-filter-re", "--filter",
            help="Apply regex filter to list (annotations).",
            nargs="+")

        parser.add_argument(
            "--id-filter-re", "--id",
            help="Apply regex filter to list (ids).",
            nargs="+")

        parser.add_argument(
            "--size", "-s",
            help="Add snapshot sizes (takes time).",
            action="store_true")

        return parser
