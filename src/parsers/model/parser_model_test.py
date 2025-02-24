from src.parsers.enums import ModelActionType
from src.parsers.interfaces import _Parser


class ParserModelTest(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Test a labelled dataset against a training.",
            name=ModelActionType.TEST,
            help="Test based on training checkpoint.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Based on this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--training-index", "-t",
            help="Based on this training (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--build-predictions", "-p",
            help="Build predictions.",
            action="store_true")

        parser.add_argument(
            "--visualise-predictions", "-v",
            help="Visualise predictions.",
            action="store_true")

        parser.add_argument(
            "--tests-num", "-n",
            help="Max test count to run (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--split", "-s",
            help="Use this split as per config (default=%(default)s).",
            choices=["all", "validation", "test", "sparse"],
            default="sparse")

        parser.add_argument(
            "--inference-only", "-i",
            help="Run inference only. Disables testing.",
            action="store_true")

        parser.add_argument(
            "--checkpoint-path", "-c",
            help="Non default checkpoint path.")

        parser.add_argument(
            "--bindings-kvp", "-b",
            help="Define bindings as kvp pairs (key:value[:int|float|str|bool]). Takes priority over JSON.",
            nargs="+")

        parser.add_argument(
            "--bindings-json", "-b2",
            help="Define bindings as a JSON string for more complex bindings.")

        return parser
