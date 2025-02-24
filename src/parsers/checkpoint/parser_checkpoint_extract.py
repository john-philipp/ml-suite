from src.parsers.enums import CheckpointActionType
from src.parsers.interfaces import _Parser


class ParserCheckpointExtract(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Extract a checkpoint including config.",
            name=CheckpointActionType.EXTRACT,
            help="Extract a checkpoint including config.")

        parser.add_argument(
            "--training-index", "-t",
            help="Based on this training (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--version", "-v",
            help="Specify a version in the format v{x}.{y}.{z}",
            required=True)

        parser.add_argument(
            "--type", "-p",
            help="Type of checkpoint (dev|test|prod)",
            default="dev",
            choices=["dev", "test", "prod"])

        return parser
