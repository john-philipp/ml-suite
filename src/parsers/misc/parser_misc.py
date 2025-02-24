from src.parsers.enums import ModeType, MetaType
from src.parsers.interfaces import _Parser
from src.parsers.misc.parser_misc_echo import ParserMiscEcho
from src.parsers.misc.parser_misc_make_docs import ParserMiscMakeDocs


class ParserMisc(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserMiscEcho,
            ParserMiscMakeDocs,
        ]

        parser = parent_parser.add_parser(
            description="Miscellaneous actions.",
            name=ModeType.MISC,
            help="Miscellaneous actions.")

        sub_parsers = parser.add_subparsers(
            dest=MetaType.ACTION,
            help="Action to take.")

        for sub_parser_cls in sub_parser_clss:
            sub_parser_cls.add_args(sub_parsers)
