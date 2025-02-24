import threading
import time

from sensor_msgs.msg import PointCloud2
from src.handlers.node.methods import translate_colour, point_cloud_msg_as_data, build_geometry, new_visualiser, \
    update_geometries, set_perspective_cad, set_perspective_top
from src.file_logger import FileLogger
from src.node.simple_node import SimpleNode, spin
from src.parsers.interfaces import _Args
from src.handlers.interfaces import _Handler


class HandlerNodeVis(_Handler):

    def __init__(self, args):
        super().__init__(args)
        self.data_index = 0
        self.geometries = {}

    class Args(_Args):
        def __init__(self, args):
            self.topics = args.topics
            self.point_size = args.point_size
            self.zoom = args.zoom

    def handle(self):
        args: HandlerNodeVis.Args = self.args
        bindings = {}
        topic_colours = self.read_topics_s(args.topics)

        file_log = FileLogger()
        file_log.log(f"Handler: {self.__class__.__name__}")
        file_log.log(f"Bindings: {bindings}")
        file_log.log(f"Topics: {topic_colours}")
        file_log.log("Handling imports...")

        import open3d

        vis_cad = new_visualiser(open3d, args.point_size)
        vis_top = new_visualiser(open3d, args.point_size)

        node = SimpleNode("node_vis")
        for topic_name, colour in topic_colours.items():
            file_log.log(f"Subscribing to topic: {topic_name}")

            def new_listener(topic_=topic_name):
                def listener(msg_):
                    file_log.log(f"Received a message on topic: {topic_}")
                    points = point_cloud_msg_as_data(msg_)
                    geometry = build_geometry(open3d, points, topic_colours[topic_])
                    self.geometries[topic_] = geometry
                return listener

            node.register_subscriber(PointCloud2, topic_name, new_listener(topic_name))
        node_thread = threading.Thread(target=lambda: spin(node, file_log.log))
        node_thread.start()

        self.update_geometries(open3d, (vis_cad, set_perspective_cad), (vis_top, set_perspective_top), zoom=args.zoom)
        node_thread.join(0)

    @classmethod
    def read_topics_s(cls, topics_s):
        topics = {}
        for topic_s in topics_s:
            topic, colour_name = topic_s.split(":")
            topics[topic] = translate_colour(colour_name)
        return topics

    def update_geometries(self, open3d, *visualisers, zoom):
        while True:
            for vis, set_perspective in visualisers:
                update_geometries(open3d, vis, self.geometries, lambda *x: set_perspective(*x, zoom=zoom))
            time.sleep(0.2)

