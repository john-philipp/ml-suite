from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelCheckAccuracy(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Check accuracy of training.",
            name=ModelActionType.CHECK_ACCURACY,
            help="Check predictions accuracy.")

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
