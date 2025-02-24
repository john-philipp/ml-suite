from src.parsers.data.parser_data_annotate import ParserDataAnnotate
from src.parsers.data.parser_data_restore import ParserDataRestore
from src.parsers.data.parser_data_save import ParserDataSave
from src.parsers.enums import ModeType, MetaType
from src.parsers.interfaces import _Parser
from src.parsers.data.parser_data_convert import ParserDataConvert
from src.parsers.data.parser_data_list import ParserDataList


class ParserData(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserDataList,
            ParserDataConvert,
            ParserDataAnnotate,
            ParserDataRestore,
            ParserDataSave,
        ]

        parser = parent_parser.add_parser(
            description="Handles any generated data, stores and loads snapshots.",
            name=ModeType.DATA,
            help="Handle data.")

        sub_parsers = parser.add_subparsers(
            dest=MetaType.ACTION,
            help="Action to take.")

        for sub_parser_cls in sub_parser_clss:
            sub_parser_cls.add_args(sub_parsers)

        return parser
