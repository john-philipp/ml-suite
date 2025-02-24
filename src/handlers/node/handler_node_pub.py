import os
import threading
import time
from time import sleep

from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2

from src.data_provider.bin_data_provider import BinDataProvider
from src.data_provider.bin_data_reader import BinDataReader
from src.data_provider.bin_msg_converter import BinMsgConverter
from src.data_provider.error import EndOfSplitError
from src.data_provider.methods import make_header
from src.data_provider.pose_data_provider import PoseDataProvider
from src.data_provider.pose_data_reader import PoseDataReader
from src.data_provider.pose_msg_converter import PoseMsgConverter
from src.jinja_yaml_loader import JinjaYamlLoader
from src.file_logger import FileLogger
from src.path_helper import PathHelper
from src.node.simple_node import SimpleNode, spin
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler
from src.toggler import Toggler


class HandlerNodePub(_Handler):

    def __init__(self, args):
        super().__init__(args)
        self.pose_closed = False
        self.bin_closed = False

    class Args(_Args):
        def __init__(self, args):
            self.dataset_index = args.dataset_index
            self.split = args.split
            self.topic_point_cloud = args.topic_point_cloud
            self.topic_pose = args.topic_pose
            self.freq = args.freq
            self.seq_range = args.seq_range
            self.manual = args.manual

    def handle(self):
        args: HandlerNodePub.Args = self.args
        bindings = {}

        seq_start, seq_end = self.read_seq_range(args.seq_range)

        # Locate dataset, file_logger, config, and config bindings.
        dataset_path = PathHelper().generated().datasets().index(args.dataset_index).path()

        file_log = FileLogger(dataset_path)
        file_log.log(f"Handler: {self.__class__.__name__}")
        file_log.log(f"Bindings: {bindings}")
        file_log.log(f"Dataset: {file_log.link(dataset_path)}")

        config_path = f"{dataset_path}/config.yml"
        if not config_path:
            raise AssertionError("No config path set.")
        if not os.path.isfile(config_path):
            raise AssertionError(f"Config path doesn't exist: {config_path}")

        file_log.keep_file(config_path)
        config_loader = JinjaYamlLoader(config_path, dict)
        config = config_loader.load(dataset_path=dataset_path, **bindings)

        # Set split from args.
        split = f"{args.split}_split"
        file_log.log(f"Using split: {split}")
        sequence_split = config["dataset"][split]
        sequence_split = [x for x in sequence_split if seq_start <= int(x) and (seq_end == -1 or int(x) <= seq_end)]

        def make_msg_provider(data_provider, msg_converter, type_, topic_, toggle=None):
            def msg_provider():
                try:
                    data, sequence_index, data_index = data_provider.get_next()
                    msg = msg_converter.convert(data, make_header())
                    if args.manual:
                        input(f"Press [Enter] to publish to topic: {topic_}\n")

                    file_log.log(f"Publishing {type_:6}: topic={topic_:20} sequence={sequence_index} data={data_index}")
                    return msg
                except EndOfSplitError:
                    # Note, this is a raise to the finish at the moment.
                    # First topic to get consumed stops all others.
                    raise KeyboardInterrupt("All data consumed.")
            return msg_provider

        toggler = Toggler()

        bin_topic = args.topic_point_cloud
        bin_data_provider = BinDataProvider(dataset_path, sequence_split, BinDataReader())
        bin_msg_provider = make_msg_provider(
            bin_data_provider, BinMsgConverter(), "bin", bin_topic, toggler.make_toggle())

        pose_topic = args.topic_pose
        pose_data_provider = PoseDataProvider(dataset_path, sequence_split, PoseDataReader())
        pose_msg_provider = make_msg_provider(
            pose_data_provider, PoseMsgConverter(), "pose", pose_topic)

        # Set up node and register publishers.
        node = SimpleNode("node_pub")
        node.register_publisher(PointCloud2, args.topic_point_cloud, bin_msg_provider, args.freq)
        node.register_publisher(PoseWithCovarianceStamped, args.topic_pose, pose_msg_provider, args.freq)

        # And publish to given topic.
        spin(node, file_log.log)

    @staticmethod
    def read_seq_range(seq_range_s, delim=":"):
        seq_start = 0
        seq_end = -1

        parts = seq_range_s.split(delim)

        try:
            seq_start = parts[0] or seq_start
            seq_end = parts[1] or seq_end
        except IndexError:
            pass

        return int(seq_start), int(seq_end)