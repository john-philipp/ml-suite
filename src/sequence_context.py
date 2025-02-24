import os

from src.file_helpers import pj, mk


class SequenceContext:
    def __init__(self, starting_sequence, sequence_interval_s_, kitti_bin_dir_f):
        self.kitti_bin_dir_f = kitti_bin_dir_f
        self.sequence_interval_s = sequence_interval_s_
        self.sequences = [starting_sequence]
        self.starting_time = None
        self.pose_out = None
        self.time_out = None
        self.start_new_sequence(init=True)
        self.added_to_sequence = 0
        self.added_in_lifetime = 0

    @property
    def sequence(self):
        return self.sequences[-1]

    @property
    def bin_dir(self):
        return self.kitti_bin_dir_f.format(sequence=self.sequence)

    @property
    def label_dir(self):
        return pj(self.bin_dir, "..", "labels")

    @property
    def pose_file(self):
        return pj(self.bin_dir, "..", "poses.txt")

    @property
    def time_file(self):
        return pj(self.bin_dir, "..", "times.txt")

    def time_since_start(self, time):
        return time - self.starting_time

    def is_new_sequence(self, time):
        assert self.starting_time is not None
        return float(time - self.starting_time) >= self.sequence_interval_s > 0

    def start_new_sequence(self, starting_time=None, init=False):
        self.starting_time = starting_time
        if not init:
            # On init this happened already.
            self.sequences.append(self.sequence + 1)
        mk(self.bin_dir)
        mk(self.label_dir)
        pose_file = self.pose_file
        if os.path.isfile(pose_file):
            os.remove(pose_file)
        self.pose_out = open(self.pose_file, "w")
        self.time_out = open(self.time_file, "w")
        self.added_to_sequence = 0

    def increment(self):
        self.added_in_lifetime += 1
        self.added_to_sequence += 1

    def close(self):
        if self.pose_out:
            self.pose_out.close()
        if self.time_out:
            self.time_out.close()
