from src.parsers.checkpoint.parser_checkpoint_extract import ParserCheckpointExtract
from src.parsers.interfaces import _Parser
from src.parsers.enums import ModeType


class ParserCheckpoint(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserCheckpointExtract
        ]

        parser = parent_parser.add_parser(
            description="Handle checkpoint actions.",
            name=ModeType.CHECKPOINT,
            help="Handle checkpoint actions.")

        sub_parsers = parser.add_subparsers(
            dest="action",
            help="Action to take.")

        for sub_parser_cls in sub_parser_clss:
            sub_parser_cls.add_args(sub_parsers)

        return parser
