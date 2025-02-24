from geometry_msgs.msg import PoseWithCovarianceStamped

from src.data_provider.interface import _IMsgConverter


class PoseMsgConverter(_IMsgConverter):
    def convert(self, data, header):
        msg = PoseWithCovarianceStamped()
        msg.header = header
        msg.pose.pose.position.x = data[0]
        msg.pose.pose.position.y = data[1]
        msg.pose.pose.position.z = data[2]
        return msg
