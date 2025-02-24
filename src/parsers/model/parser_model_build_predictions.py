from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelBuildPredictions(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Build predictions based on test data. "
                        "Requires training and testing. Viewable in point_labeler",
            name=ModelActionType.BUILD_PREDICTIONS,
            help="Build predictions viewable in point_labeler.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Based on this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--training-index", "-t",
            help="Based on this training (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        return parser
