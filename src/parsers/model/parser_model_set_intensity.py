from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelSetIntensity(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Set intensity by label.",
            name=ModelActionType.SET_INTENSITY,
            help="Set intensity by label.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Label this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--mappings", "-m",
            help="Mappings :label:intensity. (-1 == `all`, note `:` prefix)",
            required=True,
            nargs="+")

        return parser
