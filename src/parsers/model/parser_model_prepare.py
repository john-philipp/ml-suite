from src.parsers.enums import ModelActionType, ModelPrepareSparseStrategy
from src.parsers.interfaces import _Parser


class ParserModelPrepare(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Set up dataset from training. Set up sequences and splits. Requires labelled data.",
            name=ModelActionType.PREPARE,
            help="Prepare data for training.")

        parser.add_argument(
            "--dataset-index", "-d",
            help="Use this dataset (-1 == latest) (default=%(default)s).",
            default=-1,
            type=int)

        parser.add_argument(
            "--seq-end",
            help="Number of sequences to produce (max). Useful when manually "
                 "labelling and verifying workflow (default=%(default)s).",
            default=10,
            type=int)

        parser.add_argument(
            "--seq-tps",
            help="Time in seconds per sequence (with --seq-bps, use minimum) (default=%(default)s).",
            default=0.3,
            type=float)

        parser.add_argument(
            "--sparse-split-resolution", "-r",
            help="Controls the distance between bins to generate sparse split for efficient testing "
                 "(default=%(default)s).",
            default=1,
            type=float)

        parser.add_argument(
            "--sparse-split-skip",
            help="For strategy 'skip'. Skip n bins (default=%(default)s).",
            default=100,
            type=int)

        parser.add_argument(
            "--sparse-strategy",
            help="Which strategy to use to arrive at sparse split (default=%(default)s).",
            choices=ModelPrepareSparseStrategy.choices(),
            default=ModelPrepareSparseStrategy.DISTANCE)

        parser.add_argument(
            "--src-config-path",
            help="Source config path (default=%(default)s).",
            default="config/randlanet_semantickitti.yml")

        parser.add_argument(
            "--training-split",
            help="Training split to use ({train},{validation},{test}). "
                 "Label train and validation, don't label test. (default=%(default)s).",
            default="7:2:1")

        return parser
