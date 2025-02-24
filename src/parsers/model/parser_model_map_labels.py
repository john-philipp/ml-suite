from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelMapLabels(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Easily map labels of a dataset between values from the command lines.",
            name=ModelActionType.MAP_LABELS,
            help="Map labels.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Label this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--mappings", "-m",
            help="Mappings :from:to. (-1 == `all`, note `:` prefix)",
            required=True,
            nargs="+")

        return parser
