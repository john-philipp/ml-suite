from src.parsers.enums import NodeActionType
from src.parsers.interfaces import _Parser


class ParserNodePub(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Start a node to publish dataset based point-clouds.",
            name=NodeActionType.PUB,
            help="Start a node to publish dataset based point-clouds.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Use this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--split", "-s",
            help="Use this split as per config (default=%(default)s).",
            choices=["all", "validation", "test", "sparse"],
            default="sparse")

        parser.add_argument(
            "--topic-point-cloud",
            help="Topic to publish point_cloud to (default=%(default)s).",
            default="/point_cloud")

        parser.add_argument(
            "--topic-pose",
            help="Topic to publish pose to (default=%(default)s).",
            default="/pose")

        parser.add_argument(
            "--freq", "-f",
            help="Publish at this frequency (in Hz).",
            default=1.0,
            type=float)

        parser.add_argument(
            "--seq-range", "-r",
            help="Sequence range to publish (format: [start][[:end][[:step]]]).",
            default=""
        )

        parser.add_argument(
            "--manual", "-m",
            help="Publish on key-press [Enter].",
            action="store_true"
        )

        return parser
