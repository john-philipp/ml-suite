from src.parsers.node.parser_node_vis import ParserNodeVis
from src.parsers.node.parser_node_inf import ParserNodeInf
from src.parsers.node.parser_node_pub import ParserNodePub
from src.parsers.interfaces import _Parser
from src.parsers.enums import ModeType, MetaType


class ParserNode(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserNodePub,
            ParserNodeInf,
            ParserNodeVis,
        ]

        parser = parent_parser.add_parser(
            description="Node handling.",
            name=ModeType.NODE,
            help="Node handling.")

        sub_parsers = parser.add_subparsers(
            dest=MetaType.ACTION,
            help="Action to take.")

        for sub_parser_cls in sub_parser_clss:
            sub_parser_cls.add_args(sub_parsers)

        return parser
