from src.parsers.enums import NodeActionType
from src.parsers.interfaces import _Parser


class ParserNodeInf(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Start a node to listen to topic and infer based on trained checkpoint.",
            name=NodeActionType.INF,
            help="Start a node to listen to topic and infer based on trained checkpoint.")

        parser.add_argument(
            "--training-index", "-d",
            help="Use this training (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--topic-in", "-t",
            help="Topic to listen on (default=%(default)s).",
            default="/point_cloud")

        parser.add_argument(
            "--topics-out", "-o",
            help="Topic(s) to publish to (default=%(default)s).",
            default=["/inf/ground:49", "/inf/wall:51", "/inf/front:18"],
            nargs="+")

        parser.add_argument(
            "--include-labels", "-i",
            help="Publish predicted labels (for assessing accuracy)",
            action="store_true")

        parser.add_argument(
            "--visualise-predictions", "-v",
            help="Visualise predictions.",
            action="store_true")

        return parser
