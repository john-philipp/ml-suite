from src.data_provider.interface import _IMsgConverter
from src.handlers.node.methods import points_as_point_cloud_msg


class BinMsgConverter(_IMsgConverter):
    def convert(self, data, header):
        return points_as_point_cloud_msg(data, header)
