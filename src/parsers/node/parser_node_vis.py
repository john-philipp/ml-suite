from src.parsers.enums import NodeActionType
from src.parsers.interfaces import _Parser


class ParserNodeVis(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Start a node to visualise point-cloud based topics.",
            name=NodeActionType.VIS,
            help="Start a node to visualise point-cloud based topics.")

        parser.add_argument(
            "--topics", "-t",
            help="Point-cloud based topic(s) to visualise (default=%(default)s).",
            default=["/inf/ground:red", "/inf/wall:green", "/inf/front:blue"],
            nargs="+")

        parser.add_argument(
            "--point-size", "-p",
            help="Point size (default=%(default)s).",
            default=3.0,
            type=int)

        parser.add_argument(
            "--zoom", "-z",
            help="Zoom level (default=%(default)s).",
            default=0.8,
            type=float)

        return parser
