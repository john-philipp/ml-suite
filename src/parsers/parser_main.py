from src.parsers.checkpoint.parser_checkpoint import ParserCheckpoint
from src.parsers.node.parser_node import ParserNode
from src.parsers.clean.parser_clean import ParserClean
from src.parsers.enums import MetaType
from src.parsers.interfaces import _Parser
from src.parsers.data.parser_data import ParserData
from src.parsers.misc.parser_misc import ParserMisc
from src.parsers.model.parser_model import ParserModel
from src.parsers.recording.parser_recording import ParserRecording
from src.parsers.results.parser_results import ParserResults
from src.parsers.scripts.parser_scripts import ParserScripts
from src.parsers.visualise.parser_visualise import ParserVisualiser


class ParserMain(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserRecording,
            ParserData,
            ParserModel,
            ParserVisualiser,
            ParserClean,
            ParserScripts,
            ParserResults,
            ParserNode,
            ParserCheckpoint,
            ParserMisc
        ]

        parser = parent_parser.add_subparsers(
            dest=MetaType.MODE,
            help="What would you like to do?")

        for sub_parser_cls in sub_parser_clss:
            sub_parser_cls.add_args(parser)

        return parser
