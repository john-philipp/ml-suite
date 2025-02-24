from src.handlers.model.handler_model_build_predictions import HandlerModelBuildPredictions
from src.handlers.model.handler_model_label import HandlerModelLabel
from src.parsers.enums import ArchitectureType, DataFormatType, ModelType
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler
from src.methods import make_dummy
from src.tester import get_tester, TesterType


class HandlerModelTest(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.training_index = args.training_index
            self.dataset_index = args.dataset_index
            self.build_predictions = args.build_predictions
            self.visualise_predictions = args.visualise_predictions
            self.checkpoint_path = args.checkpoint_path
            self.inference_only = args.inference_only
            self.tests_num = args.tests_num
            self.split = args.split

            self.data_format = args.data_format
            self.model = args.model
            self.arch = args.arch

            self.bindings_kvp = args.bindings_kvp
            self.bindings_json = args.bindings_json

    def handle(self):
        args: HandlerModelTest.Args = self.args

        if args.arch == ArchitectureType.OPEN3D_ML:
            if args.data_format == DataFormatType.KITTI:
                if args.model == ModelType.RANDLANET:
                    # NB: This requires the correct venv.
                    # It's encouraged to keep tester envs
                    # separate to avoid a dependency mess.
                    tester = get_tester(
                        tester_type=TesterType.OPEN3D_ML_RANDLANET,
                        config_name="config.yml",
                        args=args)
                    tester.test()

                    if args.build_predictions:
                        self.build_predictions(args)

                    if args.visualise_predictions:
                        self.visualise()

                    return

        raise NotImplementedError()

    @staticmethod
    def build_predictions(args):
        handler_args = HandlerModelBuildPredictions.Args(args)
        handler = HandlerModelBuildPredictions(handler_args)
        handler.handle()

    @staticmethod
    def visualise():
        HandlerModelLabel.open_point_labeler("_generated/04-predictions", -1, make_dummy())
