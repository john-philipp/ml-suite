from src.parsers.interfaces import _Parser
from src.parsers.enums import ModeType, MetaType
from src.parsers.scripts.parser_scripts_run import ParserScriptsRun


class ParserScripts(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserScriptsRun
        ]

        parser = parent_parser.add_parser(
            description="Script handling.",
            name=ModeType.SCRIPTS,
            help="Handle scripts.")

        sub_parsers = parser.add_subparsers(
            dest=MetaType.ACTION,
            help="Action to take.")

        for sub_parser_cls in sub_parser_clss:
            sub_parser_cls.add_args(sub_parsers)

        return parser
