import time

from std_msgs.msg import Header


def make_header():
    header = Header()
    now = time.time()
    secs = int(now)
    header.stamp.sec = secs
    header.stamp.nanosec = int((now - secs) * 1e9)
    header.frame_id = "map"
    return header
