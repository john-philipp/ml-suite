from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelFeatureStats(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Check dataset feature stats.",
            name=ModelActionType.FEATURE_STATS,
            help="Check dataset feature stats.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Label this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        return parser
