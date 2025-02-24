import secrets

import rclpy
from rclpy.node import Node
rclpy.init()


class SimpleNode(Node):
    def __init__(self, name="simple_node"):
        super().__init__(name)
        self._subscribers = {}
        self._publish_timers = {}
        self._my_publishers = {}

    def register_subscriber(self, msg_type, topic_name, on_receive, qos_profile=10):
        uid = secrets.token_hex(32)
        subscriber = self.create_subscription(msg_type, topic_name, on_receive, qos_profile)
        self._subscribers[uid] = subscriber
        return uid, subscriber

    def register_publisher(self, msg_type, topic_name, msg_provider=None, freq=None, qos_profile=0):
        uid = secrets.token_hex(32)
        publisher = self.create_publisher(msg_type, topic_name, qos_profile=qos_profile)
        if freq and msg_provider:
            self._publish_timers[uid] = self.create_timer(1. / freq, lambda: publisher.publish(msg_provider()))
        self._my_publishers[uid] = publisher
        return uid, publisher


def spin(node, log):
    log("Spinning until KeyboardInterrupt...")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt as ex:
        log(f"Shutting down: {ex}")
    node.destroy_node()
