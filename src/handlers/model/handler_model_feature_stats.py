from src.handlers import _Handler
from src.parsers.enums import DataFormatType
from src.parsers.interfaces import _Args


class HandlerModelFeatureStats(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.dataset_index = args.dataset_index
            self.data_format = args.data_format

    def handle(self):
        args: HandlerModelFeatureStats.Args = self.args

        if args.data_format == DataFormatType.KITTI:
            self.handle_kitti(args)
            return

        raise NotImplementedError()

    @staticmethod
    def handle_kitti(args):
        pass
