from src.parsers.interfaces import _Parser
from src.parsers.enums import DataFormatType, ArchitectureType, ModeType, MetaType
from src.parsers.visualise.parser_visualise_dataset import ParserVisualiseDataset
from src.parsers.visualise.parser_visualise_predictions import ParserVisualisePredictions


class ParserVisualiser(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserVisualiseDataset,
            ParserVisualisePredictions,
        ]

        parser = parent_parser.add_parser(
            description="Visualise data using point_labeler.",
            name=ModeType.VISUALISE,
            help="Visualise data.")

        sub_parsers = parser.add_subparsers(
            dest=MetaType.ACTION,
            help="Action to take.")

        for sub_parser_cls in sub_parser_clss:
            sub_parser = sub_parser_cls.add_args(sub_parsers)

            sub_parser.add_argument(
                "--arch",
                help="Use this architecture (default=%(default)s).",
                choices=ArchitectureType.choices(),
                default=ArchitectureType.OPEN3D_ML)

            sub_parser.add_argument(
                "--data-format",
                help=f"Use this data format (default=%(default)s).",
                choices=DataFormatType.choices(),
                default=DataFormatType.KITTI)

        return parser
