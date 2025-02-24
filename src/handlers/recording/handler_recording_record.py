import time
import uuid

import numpy as np

from src.parsers.interfaces import _Args
from src.file_helpers import pj, rm, mk
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler
from src.methods import topic_to_dir, xyz_to_pose_array, pose_array_to_pose_line, mark_path
from src.path_handler import PathHandler
from src.this_env import GLOBALS


log = GLOBALS.log
max_intensity = 0


class HandlerRecordingRecord(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.as_str = args.as_str
            self.topic_point_cloud = args.topic_point_cloud
            self.topic_pose = args.topic_pose
            self.reference = args.reference

    def handle(self):
        args: HandlerRecordingRecord.Args = self.args

        # Import includes rclpy. So we're only importing on use.
        # Interferes somewhat with other envs.
        from src.node.simple_node import SimpleNode
        log.info(f" Will record (as_str={args.as_str}).")
        listener = SimpleNode()

        uid = uuid.uuid4().hex
        recording_path_handler = PathHandler("_generated/01-recordings")
        recording_path = recording_path_handler.get_next_path(uid[:4])
        mk(recording_path)

        file_log = FileLogger(recording_path)
        file_log.log("Starting recording...")
        file_log.log(f"Bagfile: {args.reference}")
        file_log.add_infos(
            local=True, root=True,
            args=args.__dict__,
            bagfile=args.reference)

        mark_path(recording_path, "bagfile", args.reference, truncate=None)

        if args.as_str:
            msg_path = f"{recording_path}/str"
        else:
            msg_path = f"{recording_path}/msg"

        topics = self._set_up_recording(
            [args.topic_point_cloud, args.topic_pose], msg_path, listener, args.as_str)
        max_intensity = self._handle_recording(listener)

        file_log.log("Completed recording.")
        file_log.add_infos(local=True, root=True, topics=topics, max_intensity=float(max_intensity))
        file_log.close()

    @staticmethod
    def _handle_recording(node):
        import rclpy
        log.info("Recording until KeyboardInterrupt...")
        try:
            rclpy.spin(node)
        except KeyboardInterrupt as ex:
            log.info(f" Shutting down: {ex}")
        node.destroy_node()
        global max_intensity
        log.warning(f"Max intensity was: {max_intensity}")
        return max_intensity

    def _set_up_recording(self, record_topics, msg_path, node, as_str):
        topics = {}
        for topic in record_topics:
            print(topic)
            msg_type = self._get_topic_type(topic)
            topic_dir = topic_to_dir(topic)
            msg_dir = pj(msg_path, topic_dir)
            log.info(f" Will record msgs for: topic={topic} msg_type={msg_type.__name__} msg_dir={msg_dir}")
            rm(msg_dir)
            mk(msg_dir)

            topics[topic] = {}
            topics[topic]["count"] = 0
            topics[topic]["type"] = f"{msg_type.__module__}.{msg_type.__name__}"

            def to_filename(msg):
                return self._filename_from_timestamp(msg)

            # Be careful without shadow vars in local functions or lambdas.
            # Variables are stored by reference, not value. So in a for loop,
            # the latest value of that variable will be used.
            def write_msg(msg, topic_=topic, msg_dir_=msg_dir):
                topics[topic_]["count"] += 1
                self._msg_to_file(msg, msg_dir_, topic_, to_filename, as_str)

            node.register_subscriber(msg_type, topic, write_msg)
        return topics

    @staticmethod
    def _get_topic_type(topic):
        import geometry_msgs.msg
        import sensor_msgs.msg
        return {
            "/pose": geometry_msgs.msg.PoseWithCovarianceStamped,
            "/point_cloud": sensor_msgs.msg.PointCloud2,
        }[topic]

    def _msg_to_file(self, msg, to_dir, topic, to_filename, as_str):
        filepath = pj(to_dir, to_filename(msg))
        log.info(f" Will write: {filepath}")
        if as_str:
            self._write_str_to_file(msg, filepath)
        else:
            file_writer = self._get_topic_writer(topic)
            file_writer(msg, filepath)

    @staticmethod
    def _write_pc_to_file(msg, filepath):
        import sensor_msgs_py.point_cloud2 as pc2

        points = pc2.read_points(msg, field_names=["x", "y", "z", "intensity"], skip_nans=True)
        x = points["x"]
        y = points["y"]
        z = points["z"]
        i = np.divide(points["intensity"], 255)

        # Slightly hacky.
        global max_intensity
        next_max_i = max(i)
        max_intensity = max(max_intensity, next_max_i)

        points2 = np.stack((x, y, z, i), axis=-1).astype(np.float32)
        points2.tofile(filepath + ".bin")

    def _get_topic_writer(self, topic):
        return {
            "/pose": self._write_pose_to_file,
            "/point_cloud": self._write_pc_to_file,
        }[topic]

    @staticmethod
    def _write_str_to_file(msg, filepath):
        with open(filepath + ".txt", "w") as f:
            f.write(f"{msg.__str__()}\n")

    @staticmethod
    def _write_pose_to_file(msg, filepath):
        position = msg.pose.pose.position
        pose_array = xyz_to_pose_array(position.x, position.y, position.z)
        with open(filepath + ".txt", "w") as f:
            f.write(f"{pose_array_to_pose_line(pose_array)}\n")

    @staticmethod
    def _filename_from_timestamp(msg, prefix="", suffix=""):
        timestamp = msg.header.stamp
        if not timestamp.sec:
            # Default to now.
            now = time.time()
            secs = int(now)
            timestamp.sec = secs
            timestamp.nanosec = int((now - secs) * 1e9)
        return f"{prefix}{timestamp.sec}.{timestamp.nanosec:09}{suffix}"
