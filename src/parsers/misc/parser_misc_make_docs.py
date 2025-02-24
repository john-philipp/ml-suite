from src.parsers.enums import MiscActionType
from src.parsers.interfaces import _Parser


class ParserMiscMakeDocs(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Generate simple text based docs.",
            name=MiscActionType.MAKE_DOCS,
            help="Make docs.")

        return parser
