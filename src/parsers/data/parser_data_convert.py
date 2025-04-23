from src.parsers.interfaces import _Parser
from src.parsers.enums import DataActionType, DataFormatType


class ParserDataConvert(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Convert recording to a dataset for training.",
            name=DataActionType.CONVERT,
            help="Convert to dataset.")

        parser.add_argument(
            "--recording-index", "-r",
            help=f"Recording index to use (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--data-format",
            help=f"Data format (default=%(default)s).",
            choices=DataFormatType.choices(),
            default=DataFormatType.KITTI)

        parser.add_argument(
            "--bin-topic",
            help="Use as topic for bin sequences (default=%(default)s).",
            default="/point_cloud")

        parser.add_argument(
            "--pose-topic",
            help="Use as topic for poses (default=%(default)s).",
            default="/pose")

        parser.add_argument(
            "--no-center-on-origin",
            help="Don't center poses around origin (default=%(default)s).",
            default=False,
            action="store_true")

        parser.add_argument(
            "--no-reduce",
            help="Don't reduce recording based on actual records used.",
            default=False,
            action="store_true")

        return parser
