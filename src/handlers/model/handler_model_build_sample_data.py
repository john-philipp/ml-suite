import os
import shutil
import time
import uuid

import numpy as np
from tqdm import tqdm

from src.parsers.enums import DataFormatType, ArchitectureType, ModelType
from src.parsers.interfaces import _Args
from src.jinja_yaml_loader import JinjaYamlLoader
from src.file_helpers import rm, mk
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler
from src.methods import mark_path, bindings_from_args, get_dir_size, count_labels, round_float
from src.path_handler import PathHandler
from src.sample_data_config import SampleDataConfig
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerModelBuildSampleData(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.show_frames_dt = args.show_frames_dt
            self.show_frames = args.show_frames
            self.config_path = args.config_path
            self.data_format = args.data_format
            self.model = args.model
            self.arch = args.arch

            self.bindings_kvp = args.bindings_kvp
            self.bindings_json = args.bindings_json

    def handle(self):
        args: HandlerModelBuildSampleData.Args = self.args

        if args.arch == ArchitectureType.OPEN3D_ML:
            if args.data_format == DataFormatType.KITTI:
                if args.model == ModelType.RANDLANET:
                    log.info("Will build sample data.")
                    self._handle_build_sample_data(args)
                    return

        raise NotImplementedError()

    @classmethod
    def _handle_build_sample_data(cls, args: Args):
        bindings = bindings_from_args(args.bindings_kvp, args.bindings_json)

        dataset_uid = uuid.uuid4().hex
        dataset_path_handler = PathHandler("_generated/02-datasets")
        dataset_path = dataset_path_handler.get_next_path(dataset_uid[:4])
        mk(dataset_path)

        file_log = FileLogger(dataset_path)
        file_log.log(f"Handler: {cls.__name__}")
        file_log.log(f"Bindings: {bindings}")
        file_log.keep_file(args.config_path)

        file_log.log("Handling imports...")
        from src.methods_open3d import make_point, make_fov
        import open3d as o3d

        config: SampleDataConfig = JinjaYamlLoader(
            args.config_path, lambda **kwargs: SampleDataConfig(**kwargs)).load(**bindings)

        file_log.log(f"Using config: {file_log.link(args.config_path)}")
        file_log.log("Building sample data...")

        cam_pos = np.array(config.fov_init_pos)
        cam_dir = np.array(config.fov_init_dir)
        pts = np.empty((0, 4), dtype=np.float32)
        labels_l = []

        res = config.resolution
        min_x_actual = config.min_x * res
        max_x_actual = config.max_x * res
        min_y_actual = config.min_y * res
        max_y_actual = config.max_y * res

        corner_1 = make_point(np.array([min_x_actual - 1, min_y_actual - 1, 0]), 0.1 * res)
        corner_2 = make_point(np.array([max_x_actual + 1, max_y_actual + 1, 0]), 0.1 * res)

        def r(eps_=config.epsilon):
            return eps_ * (np.random.rand() - 0.5) * res

        def add_point(x_=0, y_=0, z_=0, intensity=0, eps_=config.epsilon):
            return np.vstack([
                pts, [
                    x_ * res + r(eps_),
                    y_ * res + r(eps_),
                    z_ * res + r(eps_),
                    intensity]])

        file_log.log(f"Building objects...")
        progress = tqdm(desc="Objects", total=len(config.objects))
        for obj in config.objects:
            if obj.type == "solid-rectangle":
                for x in range(obj.args['x_0'][0], obj.args['x_1'][0] + 1):
                    for y in range(obj.args['x_0'][1], obj.args['x_1'][1] + 1):
                        for z in range(obj.args['x_0'][2], obj.args['x_1'][2] + 1):
                            pts = add_point(x_=x, y_=y, z_=z, intensity=obj.intensity)
                            labels_l.append(obj.label)
            progress.update(1)
        progress.close()

        pts_seen = np.zeros(pts.shape[0], dtype=bool)
        labels = np.array(labels_l, dtype=np.uint32)

        vis = None
        if args.show_frames:
            vis = o3d.visualization.Visualizer()
            vis.create_window()

        bin_i = 0
        seen_ratio = 0
        time_taken = 0

        config_name = "randlanet_semantickitti.yml"
        shutil.copyfile(f"config/{config_name}", f"{dataset_path}/config.yml")
        mark_path(dataset_path, "dataset", dataset_uid)
        seq_full = f"{dataset_path}/dataset/sequences/.full"

        if os.path.isdir(seq_full):
            rm(seq_full)

        seq_full_labels = f"{seq_full}/labels"
        seq_full_bins = f"{seq_full}/velodyne"
        mk(seq_full)
        mk(seq_full_labels)
        mk(seq_full_bins)
        shutil.copyfile("config/calib.txt", f"{seq_full}/calib.txt")
        poses_file = open(f"{seq_full}/poses.txt", "w")
        times_file = open(f"{seq_full}/times.txt", "w")

        file_log.log(f"Building bins using fov: (degrees={config.fov_degrees}, distance={config.fov_distance})")
        progress = tqdm(desc="Bins", total=config.bins_n)
        while bin_i < config.bins_n or seen_ratio < 0.9:
            # cam_pos += [r(30), r(30), 0]
            cam_pos[0] = r(config.grid_size)
            cam_pos[1] = r(config.grid_size)
            cam_pos[0] = max(min_x_actual + 1, float(cam_pos[0]))
            cam_pos[0] = min(max_x_actual - 1, float(cam_pos[0]))
            cam_pos[1] = max(min_y_actual + 1, float(cam_pos[1]))
            cam_pos[1] = min(max_x_actual - 1, float(cam_pos[1]))
            # cam_dir += [r(30), r(30), 0]
            cam_dir[0] = r(30)
            cam_dir[1] = r(30)

            camera, vis_pts, vis_idxs = make_fov(
                pts, cam_pos, cam_dir, config.fov_degrees, config.fov_distance, 0.5 * res)
            pts_seen |= vis_idxs
            seen_ratio = np.sum(pts_seen == True) / pts_seen.size

            if vis:
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(vis_pts)
                point_cloud.paint_uniform_color(config.pc_colour)

                vis.clear_geometries()
                vis.add_geometry(point_cloud)
                vis.add_geometry(corner_1)
                vis.add_geometry(corner_2)
                vis.add_geometry(camera)
                vis.poll_events()
                vis.update_renderer()
                time.sleep(args.show_frames_dt)

            x = cam_pos[0]
            y = cam_pos[1]
            z = cam_pos[2]
            pose_line = f"1 0 0 {x} 0 1 0 {y} 0 0 1 {z}"
            # pose_line = f"1 0 0 0 0 1 0 0 0 0 1 0"
            time_line = time_taken

            # print(f"bin={bin_i:6} seen={seen_ratio:6.3f}  time={time_line:.3f} pose={pose_line}")
            poses_file.write(f"{pose_line}\n")
            times_file.write(f"{time_line}\n")
            labels_to_write = labels[vis_idxs]

            labels_to_write.tofile(f"{seq_full_labels}/{bin_i:06}.label")
            points_to_write = pts[vis_idxs]

            # visible_points.
            # TODO can just use visible_points here.
            # TODO there's a difference here between the point_labeler and rviz.
            #   Are they using separate frames of reference? Or should I normalise the point clouds
            #   And then simple just give single position poses? Set coordinate frame for rviz?
            x = points_to_write[..., 0] - x
            y = points_to_write[..., 1] - y
            z = points_to_write[..., 2]
            i = points_to_write[..., 3]
            points2 = np.stack((x, y, z, i), axis=-1).astype(np.float32)
            points2.tofile(f"{seq_full_bins}/{bin_i:06}.bin")

            time_taken += config.dt
            bin_i += 1
            progress.update(1)

        progress.n = config.bins_n
        progress.close()

        poses_file.close()
        times_file.close()

        if vis:
            vis.destroy_window()

        file_log.log("Counting labels...")
        label_counts = count_labels(dataset_path)
        labeled_all = sum([y for x, y in label_counts.items() if x != 0])

        unlabeled = 0
        if 0 in label_counts:
            unlabeled = label_counts[0]
        labeled_pec = round_float(labeled_all / (labeled_all + unlabeled), 2)

        performed_action = dict(
            timestamp=file_log.timestamp,
            type="build-sample-data",
            dataset=file_log.link(dataset_path),
            dataset_size=get_dir_size(dataset_path),
            logs=file_log.link(file_log.get_log_dir()),
            args=args.__dict__,
            label_counts=label_counts,
            labeled=labeled_pec,
            epsilon=float(config.epsilon),
        )

        file_log.add_infos(root=True, append=True, performed_actions=performed_action)
        file_log.add_infos(local=True, performed_actions=[performed_action])
        file_log.log(f"Written to: {FileLogger.link(dataset_path)}")
        file_log.close()
