import logging
import os
import shutil
import uuid
from decimal import Decimal

from tqdm import tqdm

from src.parsers.interfaces import _Args
from src.parsers.enums import DataFormatType
from src.file_helpers import pj, rm, mk
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler
from src.methods import topic_to_dir, xyz_to_pose_array, pose_array_to_pose_line, cp_markers, mark_path, round_float, \
    get_dir_size
from src.path_handler import PathHandler
from src.sequence_context import SequenceContext
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerDataConvert(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.no_center_on_origin = args.no_center_on_origin
            self.recording_index = args.recording_index
            self.data_format = args.data_format
            self.pose_topic = args.pose_topic
            self.bin_topic = args.bin_topic
            self.reduce = not args.no_reduce
            self.rn_full = True

    def handle(self):
        args: HandlerDataConvert.Args = self.args

        recording_path_handler = PathHandler("_generated/01-recordings")
        recording_path = recording_path_handler.get_path(args.recording_index)
        recording_uid = recording_path.split("_")[-1]
        msg_path = f"{recording_path}/msg"

        dataset_uid = uuid.uuid4().hex
        dataset_path_handler = PathHandler("_generated/02-datasets")
        dataset_path = dataset_path_handler.get_next_path(suffix=dataset_uid[:4])
        mk(dataset_path)

        file_log = FileLogger(dataset_path)
        file_log.log(f"Starting handler: {self.__class__.__name__}")
        file_log.log(f"Recording:\t{file_log.link(recording_path)}")
        file_log.log(f"Dataset:\t{file_log.link(dataset_path)}")
        mark_path(dataset_path, "recording", recording_uid)

        sequences = None
        topics = dict()
        if args.data_format == DataFormatType.KITTI:
            file_log.log(f"Format:\tkitti")

            src_pose_dir = pj(msg_path, topic_to_dir(args.pose_topic))
            src_bin_dir = pj(msg_path, topic_to_dir(args.bin_topic))

            file_log.log(f"Bins src:\t{file_log.link(src_bin_dir)}")
            file_log.log(f"Pose src:\t{file_log.link(src_pose_dir)}")
            dst_dir_f = pj(dataset_path, "dataset/sequences/{sequence:02d}/velodyne").__str__()

            if not os.path.isdir(src_bin_dir):
                raise ValueError(f"Missing dir: {src_bin_dir}")

            if not os.path.isdir(src_pose_dir):
                raise ValueError(f"Missing dir: {src_pose_dir}")

            bins_with_closest_pose = self._associate_bins_with_closest_pose(src_bin_dir, src_pose_dir)

            # Remove unnecessary files.
            bins_to_keep = set()
            poses_to_keep = set()
            for bin_, pose in bins_with_closest_pose:
                bins_to_keep.add(f"{bin_:.09f}.bin")
                poses_to_keep.add(f"{pose:.09f}.txt")

            all_bins = os.listdir(src_bin_dir)
            all_poses = os.listdir(src_pose_dir)

            bins_to_rm = set(x for x in all_bins if x not in bins_to_keep)
            poses_to_rm = set(x for x in all_poses if x not in poses_to_keep)

            total_rm = len(bins_to_rm) + len(poses_to_rm)

            if args.reduce and total_rm > 0:

                def remove(path, files):
                    progress = tqdm(total=len(files))
                    for file in files:
                        file_path = pj(path, file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        progress.update(1)
                    progress.n = progress.total
                    progress.close()

                log.info("Removing unnecessary files...")
                remove(src_bin_dir, bins_to_rm)
                remove(src_pose_dir, poses_to_rm)

            file_log.log(f"Bins(rm):\t{len(all_bins)}({len(bins_to_rm)})")
            topics["bin_count"] = len(all_bins) - len(bins_to_rm)
            topics["pose_count"] = len(all_poses) - len(poses_to_rm)

            sequences = self._write_kitti_dataset(
                src_pose_dir, src_bin_dir, dst_dir_f, bins_with_closest_pose, seq_tps=0)

            file_log.log(f"Sequences:\t{len(sequences)}")

            if not args.no_center_on_origin:
                file_log.log("Recentering around origin...")
                for sequence in sequences:

                    max_xyz = [None, None, None]
                    min_xyz = [None, None, None]
                    pose_file = pj(dst_dir_f.format(sequence=sequence), "..", "poses.txt").__str__()
                    poses = []
                    with open(pose_file, "r") as f:
                        # RAM inefficient?
                        lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line:
                            parts = line.split(" ")
                            xyz = [float(parts[3]), float(parts[7]), float(parts[11])]
                            poses.append(xyz)
                            for i in range(0, 3):
                                if max_xyz[i] is None or max_xyz[i] < xyz[i]:
                                    max_xyz[i] = xyz[i]
                                if min_xyz[i] is None or min_xyz[i] > xyz[i]:
                                    min_xyz[i] = xyz[i]

                    offsets = []
                    for i in range(0, 3):
                        offsets.append((max_xyz[i] + min_xyz[i]) / 2)

                    shutil.copy(pose_file, pose_file + ".bak")

                    with open(pose_file, "w") as f:
                        for pose in poses:
                            for i in range(0, 3):
                                pose[i] -= offsets[i]
                            pose_array = xyz_to_pose_array(*pose)
                            f.write(f"{pose_array_to_pose_line(pose_array)}\n")

            # Include dummy calib.txt.
            for sequence in sequences:
                shutil.copy("config/calib.txt", f"{dst_dir_f.format(sequence=sequence)}/..")

            if args.rn_full:
                seqs_path = f"{dataset_path}/dataset/sequences"
                seq_dirs = os.listdir(seqs_path)
                full_dir = pj(seqs_path, ".full")
                rm(full_dir)

                # We distinguish between convert and training split.
                # So during convert, we just want .full at the end.
                # The training split is built separately.
                assert len(seq_dirs) == 1

                seq_dir = pj(seqs_path, seq_dirs[0])
                os.rename(seq_dir, full_dir)

        file_log.log("Done converting.")

        performed_action = dict(
            timestamp=file_log.timestamp,
            type="convert",
            recording=file_log.link(recording_path),
            recording_size=get_dir_size(recording_path),
            dataset=file_log.link(dataset_path),
            dataset_size=get_dir_size(dataset_path),
            logs=file_log.link(file_log.get_log_dir()),
            args=args.__dict__,
            labeled=round_float(0, 2),
            sequences=dict(
                count=len(sequences),
                all=sequences,
            ),
            topics=topics,
        )

        file_log.add_infos(root=True, append=True, performed_actions=performed_action)
        file_log.add_infos(local=True, performed_actions=[performed_action])
        file_log.log(f"Written to: {FileLogger.link(dataset_path)}")
        file_log.close()

    @staticmethod
    def _write_kitti_dataset(pose_dir, bin_dir, kitti_bin_dir_f, bins_with_closest_pose, seq_tps):

        log.info("Writing kitti dataset to file...")
        sequence_interval_s = seq_tps
        sequence_start = 0

        context = SequenceContext(sequence_start, sequence_interval_s, kitti_bin_dir_f)
        bins_count = len(bins_with_closest_pose)
        pbar = tqdm(total=bins_count)

        try:
            for i, (bin_time, pose_time) in enumerate(bins_with_closest_pose):

                # We just use pose time. Bin time is close.
                if context.starting_time is None:
                    context.starting_time = pose_time

                pbar.update(1)

                bin_file_dst = os.path.join(context.bin_dir, f"{i:06}.bin")
                pose_file_src = os.path.join(pose_dir, f"{pose_time:.09f}")
                bin_file_src = os.path.join(bin_dir, f"{bin_time:.09f}")

                with open(pose_file_src + ".txt", "r") as p_in:
                    context.pose_out.write(p_in.readline())

                context.time_out.write(f"{context.time_since_start(pose_time).__float__()}\n")
                shutil.copy(bin_file_src + ".bin", bin_file_dst)

        finally:
            context.close()
            pbar.n = pbar.total
            pbar.close()

        return context.sequences

    @staticmethod
    def _associate_bins_with_closest_pose(raw_bin_dir, raw_pose_dir):
        log.info(" Associating bins with closest pose...")
        bins = [Decimal(x.replace(".bin", "")) for x in os.listdir(raw_bin_dir)]
        poses = [Decimal(x.replace(".txt", "")) for x in os.listdir(raw_pose_dir)]
        bins.sort()
        poses.sort()

        # For each bin. Look for pose closest.
        # First pose should be less than first bin.
        # Find first bin lower than first pose.
        bin_start = 0
        while True:
            try:
                if bins[bin_start + 1] >= poses[0]:
                    break
            except IndexError:
                raise ValueError("Couldn't find any bins later than first pose.")
            bin_start += 1

        bins_with_closest_pose = []
        offset = 0
        bins_count = len(bins)
        pbar = tqdm(total=bins_count)
        relevant_poses = {}
        bin_ = None
        for i, bin_ in enumerate(bins[bin_start:]):

            relevant_poses.clear()
            last_diff = None
            for j, pose in enumerate(poses[offset:]):
                diff = abs(bin_ - pose)
                if last_diff is None or diff <= last_diff:
                    relevant_poses[diff] = (j, pose)
                    last_diff = diff
                else:
                    min_key = min(relevant_poses.keys())
                    min_j, min_pose = relevant_poses[min_key]
                    bins_with_closest_pose.append((bin_, min_pose))
                    offset = min_j + 1
                    break

            pbar.update(1)

        if bin_ and len(relevant_poses) > 0:
            min_key = min(relevant_poses.keys())
            min_j, min_pose = relevant_poses[min_key]
            bins_with_closest_pose.append((bin_, min_pose))

        pbar.n = pbar.total
        pbar.close()
        return bins_with_closest_pose
