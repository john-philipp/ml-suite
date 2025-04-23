from src.handlers.model.handler_model_label import HandlerModelLabel
from src.parsers.enums import DataFormatType, ArchitectureType
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler


class HandlerVisualiseDataset(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.config_name = "config.yml"
            self.dataset_index = args.dataset_index
            self.screenshot_config = args.screenshot_config
            self.write_screenshot_config = args.write_screenshot_config
            self.take_screenshot = args.take_screenshot
            self.data_format = args.data_format
            self.arch = args.arch

    def handle(self):
        args: HandlerVisualiseDataset.Args = self.args

        if args.data_format == DataFormatType.KITTI:
            if args.arch == ArchitectureType.OPEN3D_ML:
                HandlerModelLabel.open_point_labeler("_generated/02-datasets", args.dataset_index, args)
                return

        raise NotImplementedError()
