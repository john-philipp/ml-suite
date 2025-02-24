import open3d as o3d
import numpy as np


def make_point(pos, radius, colour=None):
    pt = o3d.geometry.TriangleMesh().create_sphere(radius=radius)
    pt.paint_uniform_color(colour or [1, 0, 0])
    pt.translate(pos)
    return pt


def make_fov(points, cam_pos, cam_dir, fov_deg, max_dist, radius):
    camera_pt = make_point(cam_pos, radius)

    fov_radians = np.radians(fov_deg)
    direction_vectors = points[:, :3] - cam_pos
    direction_vectors_normalized = direction_vectors / np.linalg.norm(direction_vectors, axis=1)[:, np.newaxis]

    view_direction = cam_dir - cam_pos
    view_direction_normalized = view_direction / np.linalg.norm(view_direction)
    dot_products = np.clip(np.dot(direction_vectors_normalized, view_direction_normalized), -1.0, 1.0)

    angles = np.arccos(dot_products)
    visible_idxs_angles = angles <= (fov_radians / 2)

    distances = np.linalg.norm(direction_vectors, axis=1)
    visible_idxs_distances = distances <= max_dist

    visible_idxs = visible_idxs_angles & visible_idxs_distances
    visible_pts = points[visible_idxs][:, :3]
    return camera_pt, visible_pts, visible_idxs


