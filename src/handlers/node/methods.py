from sensor_msgs.msg import PointCloud2, PointField

import sensor_msgs_py.point_cloud2 as pc2
import numpy as np
from std_msgs.msg import Header


def point_cloud_msg_as_data(msg: PointCloud2):
    points = pc2.read_points(msg, field_names=["x", "y", "z", "intensity"], skip_nans=True)
    x = points["x"]
    y = points["y"]
    z = points["z"]
    i = points["intensity"]
    return np.stack((x, y, z, i), axis=-1).astype(np.float32)


def data_as_point_cloud_msg(data, header=None):
    intensity = 255 * data["feat"]
    points = np.hstack([data["point"], intensity])
    return points_as_point_cloud_msg(points, header)


def points_as_point_cloud_msg(points, header=None):
    # Define the fields
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    if not header:
        header = Header()
        header.frame_id = "map"

    return pc2.create_cloud(header, fields, points)


def new_visualiser(open3d, point_size):
    vis = open3d.visualization.Visualizer()
    vis.create_window()

    # Adjust render options for point size
    render_option = vis.get_render_option()
    render_option.point_size = point_size  # Set point size (default is 5.0)
    return vis


def build_geometry(open3d, points, colour):
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(points[:, :3])
    point_cloud.paint_uniform_color(colour)
    return point_cloud


def translate_colour(colour_name):
    colour_map = {
        "red": [1, 0, 0],
        "green": [0, 1, 0],
        "blue": [0, 0, 1],
        "yellow": [1, 1, 0],
        "cyan": [0, 1, 1],
        "magenta": [1, 0, 1],
        "black": [0, 0, 0],
    }
    try:
        return colour_map.get(colour_name.lower())
    except KeyError:
        raise ValueError(f"Found {colour_name}. Colour must be one of: {colour_map.keys()}")


def update_geometries(open3d, vis, geometries, set_perspective):
    vis.clear_geometries()

    points = None
    for topic, geometry in geometries.items():
        vis.add_geometry(geometry)
        if points is None:
            points = geometry.points
        else:
            points = np.vstack([points, geometry.points])

    if points is not None and not isinstance(points, open3d.utility.Vector3dVector):
        centroid = points.mean(axis=0)[:3]
    else:
        centroid = [0.5, 0.5, 0.5]

    set_perspective(vis, centroid)
    vis.poll_events()
    vis.update_renderer()


def set_perspective_cad(vis, centroid, zoom):
    # Set camera angle
    view_control = vis.get_view_control()
    view_control.set_front([1, 1, 1])
    view_control.set_lookat(centroid)
    view_control.set_up([0, 0, 1])
    view_control.set_zoom(zoom)


def set_perspective_top(vis, centroid, zoom):
    # Set camera angle
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_lookat(centroid)
    view_control.set_up([0, -1, 0])
    view_control.set_zoom(zoom)
