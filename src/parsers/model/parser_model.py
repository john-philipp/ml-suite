from src.parsers.interfaces import _Parser
from src.parsers.enums import DataFormatType, ArchitectureType, ModelType, ModeType, MetaType
from src.parsers.model.parser_model_calculate_weights import ParserModelCalculateWeights
from src.parsers.model.parser_model_build_predictions import ParserModelBuildPredictions
from src.parsers.model.parser_model_build_sample_data import ParserModelBuildSampleData
from src.parsers.model.parser_model_check_accuracy import ParserModelCheckAccuracy
from src.parsers.model.parser_model_feature_stats import ParserModelFeatureStats
from src.parsers.model.parser_model_set_intensity import ParserModelSetIntensity
from src.parsers.model.parser_model_label import ParserModelLabel
from src.parsers.model.parser_model_map_labels import ParserModelMapLabels
from src.parsers.model.parser_model_prepare import ParserModelPrepare
from src.parsers.model.parser_model_test import ParserModelTest
from src.parsers.model.parser_model_train import ParserModelTrain


class ParserModel(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        sub_parser_clss = [
            ParserModelBuildSampleData,
            ParserModelLabel,
            ParserModelMapLabels,
            ParserModelSetIntensity,
            ParserModelCalculateWeights,
            ParserModelFeatureStats,
            ParserModelPrepare,
            ParserModelTrain,
            ParserModelTest,
            ParserModelBuildPredictions,
            ParserModelCheckAccuracy,
        ]

        parser = parent_parser.add_parser(
            description="Handle anything model related (training/testing/labels/...).",
            name=ModeType.MODEL,
            help="Handle models.")

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

            sub_parser.add_argument(
                "--model",
                help=f"Use this model (default=%(default)s).",
                choices=ModelType.choices(),
                default=ModelType.RANDLANET)

        return parser
