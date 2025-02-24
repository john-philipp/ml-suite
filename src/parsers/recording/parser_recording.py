from src.parsers.interfaces import _Parser
from src.parsers.enums import ModeType, MetaType
from src.parsers.recording.parser_recording_record import ParserRecordingRecord


class ParserRecording(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserRecordingRecord,
        ]

        parser = parent_parser.add_parser(
            description="Record messages to convert to dataset(s) for training.",
            name=ModeType.RECORDING,
            help="Record topic messages.")

        sub_parsers = parser.add_subparsers(
            dest=MetaType.ACTION,
            help="Action to take.")

        for sub_parser_cls in sub_parser_clss:
            sub_parser_cls.add_args(sub_parsers)

        return parser
