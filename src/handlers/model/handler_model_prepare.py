import os
import random
import shutil
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from src.path_helper import PathHelperConfig, PathHelper
from src.parsers.enums import ArchitectureType, DataFormatType, ModelType, ModelPrepareSparseStrategy
from src.parsers.interfaces import _Args
from src.file_helpers import pj, rm, mk
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler
from src.methods import read_yaml, set_data, to_config_path, write_config_data, read_label_data, count_labels_2, \
    get_data, get_dir_size, estimate_split_points
from src.path_handler import PathHandler
from src.sequence_context import SequenceContext
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerModelPrepare(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.dataset_index = args.dataset_index
            self.randomise = True

            self.data_format = args.data_format
            self.model = args.model
            self.arch = args.arch

            self.sparse_split_skip = args.sparse_split_skip
            self.sparse_strategy = args.sparse_strategy

            self.src_config_path = args.src_config_path
            self.training_split = args.training_split
            self.seq_tps = args.seq_tps
            self.seq_end = args.seq_end

            self.sparse_split_resolution = args.sparse_split_resolution

    def handle(self):
        args: HandlerModelPrepare.Args = self.args

        if args.arch == ArchitectureType.OPEN3D_ML:
            if args.data_format == DataFormatType.KITTI:
                if args.model == ModelType.RANDLANET:
                    self._prepare_open3d_ml_kitti_randlanet(
                        args.dataset_index, args.src_config_path,
                        args.training_split, args.seq_end, args.seq_tps,
                        args.randomise, args)
                    return

        raise NotImplementedError()

    def _prepare_open3d_ml_kitti_randlanet(
            self, dataset_index, src_config_path, training_split, seq_end, seq_tps, randomise, args):

        # Ensure all labels defined in label file.
        config_data = read_yaml(src_config_path)

        dataset_path_handler = PathHandler("_generated/02-datasets")
        dataset_path = dataset_path_handler.get_path(dataset_index)

        file_log = FileLogger(dataset_path)
        file_log.log(f"Handler: {self.__class__.__name__}")
        file_log.log(f"Dataset: {file_log.link(dataset_path)}")
        file_log.add_infos(local=True, args=args.__dict__)

        seqs_dir = pj(dataset_path, "dataset/sequences")
        bins_dir_full = pj(seqs_dir, ".full/velodyne")
        bins = os.listdir(bins_dir_full)

        # Use minimum if defined.
        bins_n = len(bins)
        if bins_n > seq_end > 0:
            bins_n = seq_end

        # Want to remove existing sequences (ignoring .full).
        seqs = os.listdir(seqs_dir)
        seqs.remove(".full")
        seqs.sort()

        for seq in seqs:
            rm(pj(seqs_dir, seq))

        sequences, weights = self._handle_convert(dataset_path, bins_dir_full, seq_end, seq_tps, file_log)
        split_train, split_valid, split_test = self._unpack_training_split_s(training_split, len(sequences))
        if args.sparse_strategy == ModelPrepareSparseStrategy.DISTANCE:
            self._build_sparse_split_distance(dataset_index, file_log, resolution=args.sparse_split_resolution)
        elif args.sparse_strategy == ModelPrepareSparseStrategy.SKIP:
            self._build_sparse_split_skip(dataset_index, file_log, skip=args.sparse_split_skip)

        # Sanity validation.
        base_msg = "Not enough bins or too low --seq-end for training_split"

        if split_train == 0:
            raise ValueError(f"{base_msg}: train={split_train} bins_n={bins_n}")
        if split_valid == 0:
            raise ValueError(f"{base_msg}: valid={split_valid} bins_n={bins_n}")
        if split_test == 0:
            raise ValueError(f"{base_msg}: test={split_test} bins_n={bins_n}")

        def split2(container):
            container.sort()
            return [f"{x:02}" for x in container]

        if randomise:
            file_log.log("Randomising sequences.")

        training_split = []
        validation_split = []
        sparse_split = [".sparse"]
        full_split = [".full"]
        test_split = []
        all_split = []

        available = [i for i in range(len(sequences))]
        while available:
            choice_i = 0
            if randomise:
                choice_i = random.randint(0, len(available) - 1)
            choice = available.pop(choice_i)
            if len(training_split) < split_train:
                training_split.append(choice)
            elif len(validation_split) < split_valid:
                validation_split.append(choice)
            elif len(test_split) < split_test:
                test_split.append(choice)
            else:
                training_split.append(choice)
            all_split.append(choice)

        training_split = split2(training_split)
        validation_split = split2(validation_split)
        test_split = split2(test_split)
        all_split = split2(all_split)

        set_data(config_data, full_split, "dataset.full_split")
        set_data(config_data, sparse_split, "dataset.sparse_split")

        set_data(config_data, training_split, "dataset.training_split")
        set_data(config_data, validation_split, "dataset.validation_split")
        # set_data(config_data, ['00'], "dataset.validation_split")
        set_data(config_data, test_split, "dataset.test_split")
        set_data(config_data, all_split, "dataset.all_split")
        set_data(config_data, weights, "dataset.class_weights")
        default_points = get_data(config_data, "model.num_points")
        default_neighbors = get_data(config_data, "model.num_neighbors")
        set_data(config_data, "{{points}}", "model.num_points")
        set_data(config_data, "{{neighbors}}", "model.num_neighbors")

        bindings = dict(
            points=default_points,
            neighbors=default_neighbors,
        )

        config_path = to_config_path(dataset_path)
        write_config_data(config_path, bindings, config_data)
        file_log.keep_file(config_path)

        file_log.log(f"Generated {len(sequences)} sequences.")
        file_log.add_infos(root=True, )

        performed_action = dict(
            timestamp=file_log.timestamp,
            type="prepare",
            dataset=file_log.link(dataset_path),
            dataset_size=get_dir_size(dataset_path),
            logs=file_log.link(file_log.get_log_dir()),
            args=args.__dict__,
            sequence_config=dict(
                count=len(sequences),
                split_training=dict(
                    split_len=len(training_split),
                    split_pts=estimate_split_points(dataset_path, training_split)
                ),
                split_validation=dict(
                    split_len=len(validation_split),
                    split_pts=estimate_split_points(dataset_path, validation_split)
                ),
                split_test=dict(
                    split_len=len(test_split),
                    split_pts=estimate_split_points(dataset_path, test_split)
                ),
                split_all=dict(
                    split_len=len(all_split),
                    split_pts=estimate_split_points(dataset_path, all_split)
                )
            )
        )

        file_log.add_infos(root=True, append=True, performed_actions=performed_action)
        file_log.add_infos(local=True, performed_actions=[performed_action])
        file_log.log(f"Written to: {FileLogger.link(dataset_path)}")
        file_log.close()

    @staticmethod
    def _handle_convert(dataset_path, bins_dir_full, seq_end, seq_tps, file_log):
        dst_dir_f = pj(dataset_path, "dataset/sequences/{sequence:02d}/velodyne").__str__()

        sequence_interval_s = seq_tps
        sequence_start = 0
        sequence_end = seq_end - 1

        context = SequenceContext(sequence_start, sequence_interval_s, dst_dir_f)
        bins_times_full = pj(bins_dir_full, "../times.txt")
        bins_poses_full = pj(bins_dir_full, "../poses.txt")
        with open(bins_times_full, "r") as f:
            times = f.read().split("\n")
        with open(bins_poses_full, "r") as f:
            poses = f.read().split("\n")

        labels_dir_full = pj(bins_dir_full, "../labels")
        if not os.path.isdir(labels_dir_full):
            labels = []
        else:
            labels = os.listdir(labels_dir_full)
            labels.sort()

        bins = os.listdir(bins_dir_full)
        bins.sort()

        # Ignore empty lines.
        bins = [x for x in bins if x]
        poses = [x for x in poses if x]
        times = [x for x in times if x]
        labels = [x for x in labels if x]

        # Sanity check.
        if not len(bins) == len(poses) == len(times) == len(labels):
            file_log.log("Did you label, yet?")
            file_log.close()
            raise AssertionError("Did you label, yet?")

        bins_count = len(bins)
        pbar = tqdm(total=bins_count)

        label_counters = defaultdict(int)
        try:
            for i, time_ in enumerate(times):
                time_ = float(time_)

                # We just use pose time. Bin time is close.
                if context.starting_time is None:
                    context.starting_time = time_
                elif context.is_new_sequence(time_):
                    if sequence_end > 0:
                        if context.sequence >= sequence_end:
                            break
                    context.start_new_sequence(starting_time=time_)

                offset = i - context.added_to_sequence
                bin_file_dst = pj(context.bin_dir, f"{i - offset:06}.bin")
                label_file_dst = pj(context.label_dir, f"{i - offset:06}.label")
                context.pose_out.write(f"{poses[i]}\n")

                bin_file_src = pj(bins_dir_full, bins[i])
                label_file_src = pj(labels_dir_full, labels[i])
                context.time_out.write(f"{context.time_since_start(time_).__float__()}\n")

                shutil.copy(bin_file_src, bin_file_dst)
                shutil.copy(label_file_src, label_file_dst)

                label_data = read_label_data(label_file_dst)
                count_labels_2(label_data, label_counters)

                context.increment()
                pbar.update(1)

        finally:
            context.close()
            pbar.n = pbar.total
            pbar.close()

        # Include dummy calib.txt.
        for sequence in context.sequences:
            shutil.copy("config/calib.txt", f"{dst_dir_f.format(sequence=sequence)}/..")

        return context.sequences, []  # weights

    @staticmethod
    def _unpack_training_split_s(training_split_s, bins_n):
        parts = training_split_s.split(":")
        train, valid, test = int(parts[0]), int(parts[1]), int(parts[2])

        total = train + valid + test
        train /= total
        valid /= total
        test /= total

        if train <= 0:
            raise ValueError("Training split invalid: train <= 0.")
        if valid <= 0:
            raise ValueError("Training split invalid: valid <= 0.")
        if test <= 0:
            raise ValueError("Training split invalid: test <= 0.")
        if not 0.6 <= train <= 0.8:
            log.warning(f"Training split not in healthy range (0.6 <= train <= 0.8): train={train}")
        if not 0.1 <= valid <= 0.2:
            log.warning(f"Training split not in healthy range (0.1 <= valid <= 0.2): valid={valid}")
        if not 0.1 <= test <= 0.2:
            log.warning(f"Training split not in healthy range (0.1 <= test <= 0.2): test={test}")
        return int(train * bins_n), int(valid * bins_n), int(test * bins_n)

    @staticmethod
    def _build_sparse_split_distance(database_index, file_log, resolution=5e-3):
        file_log.log(f"Building sparse split: resolution={resolution}")

        ph_config = PathHelperConfig(auto_rollback=True)
        full_seq_paths = PathHelper(config=ph_config).sequence2(database_index, -1).commit()

        full_seq_path = full_seq_paths.path()
        full_poses_path = full_seq_paths.poses(assert_exists=True).path()
        full_times_path = full_seq_paths.times(assert_exists=True).path()

        # Read poses.txt.
        # Keep track of points. Calculate distance to some prev set. Ensure no point x meters in vicinity.
        with open(full_poses_path, "r") as f:
            poses_lines = f.readlines()

        poses = []
        for line in poses_lines:
            parts = line.strip().split(" ")
            x = float(parts[3])
            y = float(parts[7])
            z = float(parts[11])

            poses.append([x, y, z])

        np_poses = np.array(poses, dtype=np.float32)
        points = np_poses

        filtered_points = []
        while len(points) > 1:
            point = points[0]
            filtered_points.append(point)
            distances = np.linalg.norm(points - point, axis=1)
            points = points[distances >= resolution]
            print(f"Mean distance: {np.mean(distances):.3f} (remaining={len(points)})")

        indexes = []
        for i, p in enumerate(np_poses):
            if any(np.array_equal(p, q) for q in filtered_points):
                indexes.append(i)

        file_log.log(f"Sparse split: reduced from {len(np_poses)} to {len(indexes)}")

        # Building _test sequence.
        sparse_seq_paths = PathHelper(full_seq_path, "../.sparse", config=ph_config)

        sparse_seq_path = sparse_seq_paths.path()
        sparse_bins_paths = sparse_seq_paths.cp().bins().commit()
        sparse_labels_paths = sparse_seq_paths.cp().labels().commit()

        rm(sparse_seq_path)
        mk(sparse_seq_path)
        mk(sparse_bins_paths.path())
        mk(sparse_labels_paths.path())

        # Get bins.
        full_bins_paths = full_seq_paths.cp().bins().commit()
        full_bins = os.listdir(full_bins_paths.path())
        full_bins.sort()

        full_bins_to_keep = [x for i, x in enumerate(full_bins) if i in indexes]
        for i, bin_name in enumerate(full_bins_to_keep):
            shutil.copyfile(full_bins_paths.push(bin_name).path(), sparse_seq_paths.bins().bin(i).path())

        # Get labels.
        full_labels_paths = full_seq_paths.cp().labels().commit()
        full_labels = os.listdir(full_labels_paths.path())
        full_labels.sort()

        full_labels_to_keep = [x for i, x in enumerate(full_labels) if i in indexes]
        for i, label_name in enumerate(full_labels_to_keep):
            shutil.copyfile(full_labels_paths.push(label_name).path(), sparse_seq_paths.labels().label(i).path())

        # Reduce poses.txt, times.txt.
        sparse_poses_path = sparse_seq_paths.poses().path()
        sparse_times_path = sparse_seq_paths.times().path()

        with open(sparse_poses_path, "w") as f:
            for i, pose_line in enumerate(poses_lines):
                if i in indexes:
                    f.write(pose_line)

        with open(full_times_path, "r") as f:
            times_lines = f.readlines()

        with open(sparse_times_path, "w") as f:
            for i, time_line in enumerate(times_lines):
                if i in indexes:
                    f.write(time_line)

        # Copy calib.txt. Instances.txt.
        shutil.copyfile(full_seq_paths.calib().path(), sparse_seq_paths.calib().path())
        try:
            shutil.copyfile(full_seq_paths.instances().path(), sparse_seq_paths.instances().path())
        except FileNotFoundError:
            pass
        file_log.log("Done building sparse split.")

    @staticmethod
    def _build_sparse_split_skip(database_index, file_log, skip=100):
        file_log.log(f"Building sparse split: skip={skip}")

        ph_config = PathHelperConfig(auto_rollback=True)
        full_seq_paths = PathHelper(config=ph_config).sequence2(database_index, -1).commit()

        full_seq_path = full_seq_paths.path()
        full_poses_path = full_seq_paths.poses(assert_exists=True).path()
        full_times_path = full_seq_paths.times(assert_exists=True).path()

        # Read poses.txt.
        # Keep track of points. Calculate distance to some prev set. Ensure no point x meters in vicinity.
        with open(full_poses_path, "r") as f:
            poses_lines = f.readlines()

        file_log.log(f"Sparse split: reduced from {len(poses_lines)} to {len(poses_lines[::skip])}")

        # Building _test sequence.
        sparse_seq_paths = PathHelper(full_seq_path, "../.sparse", config=ph_config)

        sparse_seq_path = sparse_seq_paths.path()
        sparse_bins_paths = sparse_seq_paths.cp().bins().commit()
        sparse_labels_paths = sparse_seq_paths.cp().labels().commit()

        rm(sparse_seq_path)
        mk(sparse_seq_path)
        mk(sparse_bins_paths.path())
        mk(sparse_labels_paths.path())

        # Get bins.
        full_bins_paths = full_seq_paths.cp().bins().commit()
        full_bins = os.listdir(full_bins_paths.path())
        full_bins.sort()

        full_bins_to_keep = [x for _, x in enumerate(full_bins[::skip])]
        for i, bin_name in enumerate(full_bins_to_keep):
            shutil.copyfile(full_bins_paths.push(bin_name).path(), sparse_seq_paths.bins().bin(i).path())

        # Get labels.
        full_labels_paths = full_seq_paths.cp().labels().commit()
        full_labels = os.listdir(full_labels_paths.path())
        full_labels.sort()

        full_labels_to_keep = [x for _, x in enumerate(full_labels[::skip])]
        for i, label_name in enumerate(full_labels_to_keep):
            shutil.copyfile(full_labels_paths.push(label_name).path(), sparse_seq_paths.labels().label(i).path())

        # Reduce poses.txt, times.txt.
        sparse_poses_path = sparse_seq_paths.poses().path()
        sparse_times_path = sparse_seq_paths.times().path()

        with open(sparse_poses_path, "w") as f:
            for pose_line in poses_lines[::skip]:
                f.write(pose_line)

        with open(full_times_path, "r") as f:
            times_lines = f.readlines()

        with open(sparse_times_path, "w") as f:
            for time_line in times_lines[::skip]:
                f.write(time_line)

        # Copy calib.txt. Instances.txt.
        shutil.copyfile(full_seq_paths.calib().path(), sparse_seq_paths.calib().path())
        try:
            shutil.copyfile(full_seq_paths.instances().path(), sparse_seq_paths.instances().path())
        except FileNotFoundError:
            pass
        file_log.log("Done building sparse split.")
