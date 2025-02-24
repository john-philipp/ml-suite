from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelCalculateWeights(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Calculate weights to use in training.",
            name=ModelActionType.CALCULATE_WEIGHTS,
            help="Calculate weights to use in training.")

        parser.add_argument(
            "--dataset-indexes", "-d",
            help="Calculate weights based on these datasets (: == all). Syntax [{start}:{end}]. "
                 "If either limit is empty we assume range min/max. Repeatable arg. "
                 "(default=%(default)s).",
            default=[":"],
            nargs="+")

        return parser
