from src.parsers.enums import CleanActionType
from src.parsers.interfaces import _Parser


class ParserCleanDatasets(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Clean dataset(s).",
            name=CleanActionType.DATASETS,
            help="Clean dataset(s).")

        parser.add_argument(
            "--index", "-i",
            help="Clean this index (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--all", "-a",
            help="Clean all.",
            action="store_true")

        return parser
