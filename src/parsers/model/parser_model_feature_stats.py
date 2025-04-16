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
            help="Analyse this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--sequence-index", "-s",
            help="Analyse this sequence (-1 == .full, -2 == .sparse) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--feature-dims", "-f",
            help="Feature dims in data.",
            default=1,
            type=int)

        parser.add_argument(
            "--coord-dims", "-c",
            help="Coord dims in data.",
            default=3,
            type=int)

        parser.add_argument(
            "--live-plot", "-p",
            help="Display plot.",
            action="store_true")

        parser.add_argument(
            "--threads", "-t",
            help="Use n threads.",
            default=8,
            type=int)

        return parser
