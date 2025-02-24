from src.file_logger import FileLogger
from src.handlers.model.handler_model_label import HandlerModelLabel
from src.path_helper import PathHelper
from src.parsers.enums import DataFormatType, ArchitectureType
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler


class HandlerVisualisePredictions(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.config_name = "config.yml"
            self.predictions_index = args.predictions_index
            self.screenshot_config = args.screenshot_config
            self.write_screenshot_config = args.write_screenshot_config
            self.take_screenshot = args.take_screenshot
            self.data_format = args.data_format
            self.arch = args.arch

    def handle(self):
        args: HandlerVisualisePredictions.Args = self.args

        dataset_path = PathHelper().generated().predictions().index(args.predictions_index).path()
        file_log = FileLogger(dataset_path)
        file_log.log(f"Handler: {self.__class__.__name__}")
        file_log.log(f"Dataset: {file_log.link(dataset_path)}")

        if args.data_format == DataFormatType.KITTI:
            if args.arch == ArchitectureType.OPEN3D_ML:
                HandlerModelLabel.open_point_labeler(
                    "_generated/04-predictions", args.predictions_index, args, file_log)

        file_log.close()
