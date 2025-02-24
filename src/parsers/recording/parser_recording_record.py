from src.parsers.enums import RecordingActionType
from src.parsers.interfaces import _Parser


class ParserRecordingRecord(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Record point-clouds and position messages via ROS2 topics.",
            name=RecordingActionType.RECORD,
            help="Record topic messages.")

        parser.add_argument(
            "--reference", "-r",
            help="Reference for recording.",
            required=True)

        parser.add_argument(
            "--topic-point-cloud",
            help="Topic to record point_cloud (default=%(default)s).",
            default="/point_cloud")

        parser.add_argument(
            "--topic-pose",
            help="Topic to record pose (default=%(default)s).",
            default="/pose")

        parser.add_argument(
            "--as-str",
            help="Write to file as string utilising __str__().",
            action="store_true")

        return parser
