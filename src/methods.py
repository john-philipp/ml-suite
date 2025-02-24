import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from time import sleep

import humanize
import numpy as np
from ruamel.yaml import YAML
from tqdm import tqdm

from src.jinja_yaml_loader import JinjaYamlLoader
from src.path_helper import PathHelper
from src.this_env import GLOBALS


_log = GLOBALS.log
REGEX_RESPONSE_TYPES = "[a-zA-Z_.0-9]+\\("
REGEX_RESPONSE_NANS = "[= ,](nan)[,\\])]"


def topic_to_dir(topic):
    if topic.startswith("/"):
        topic = topic[1:]
    return f"topic.{topic.replace('/', '-')}"


def read_yaml(path):
    with open(path, "r") as f:
        return YAML().load(f)


def to_config_path(path_kitti):
    return f"{path_kitti}/config.yml"


def to_seq_path(seqs_dir, i_seq):
    return f"{seqs_dir}/{i_seq:02}"


def to_labels_path(seqs_dir, i_seq):
    return f"{to_seq_path(seqs_dir, i_seq)}/labels"


def to_label_path(seqs_dir, i_seq, i_label):
    return f"{to_labels_path(seqs_dir, i_seq)}/{i_label:06}.label"


def to_bins_path(seqs_dir, i_seq):
    return f"{to_seq_path(seqs_dir, i_seq)}/velodyne"


def read_label_data(path):
    return np.fromfile(path, dtype=np.uint32)


def read_bin_data(path):
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32)])
    return np.fromfile(path, dtype=dtype)


def get_timestamp():
    return (datetime.now().__str__()
            .replace(" ", "_")
            .replace(":", "")
            .replace("-", "")
            .replace(".", "-")[:-3])


def write_config_data(path, bindings, data):
    with open(path, "w") as f:
        yml = YAML()
        yml.default_flow_style = True
        f.write("# BINDINGS:\n")
        yml.dump(bindings, f)
        f.write("---\n")
        yml.dump(data, f)


def file_path(dir_name, file_name='', **format_args):
    return os.path.join(dir_name, file_name.format(**format_args)).__str__()


def pose_array_to_pose_line(pose_array):
    line = ""
    for value in pose_array:
        line += f"{value} "
    return line.strip()


def xyz_to_pose_array(x, y, z):
    return [1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z]


def _read_metadata(dir_path):
    metadata_path = os.path.join(dir_path, "metadata.yml")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r") as f:
        data = f.read()
    if not data:
        return {}
    return json.loads(data)


def _write_metadata(path, metadata_d):
    with open(os.path.join(path, "metadata.yml"), "w") as f:
        f.write(json.dumps(metadata_d, indent=2))


def get_data(data, *keys, raise_on_missing=False, default=None):
    if not keys:
        return data
    keys = ".".join(keys).split(".")
    value = data[keys[0]]
    for key in keys[1:]:
        if key not in value:
            if raise_on_missing:
                raise KeyError(key)
            return default
        value = value[key]
    return value


def set_data(data, value, *keys, append=False):
    if not keys:
        return
    keys = ".".join(keys).split(".")
    if len(keys) == 1:
        if not append:
            data[keys[0]] = value
        else:
            key = keys[0]
            if key not in data:
                data[key] = []
            container = data[key]
            assert isinstance(container, list)
            container.append(value)
    else:
        value_d = data.setdefault(keys[0], {})
        for key in keys[1:-1]:
            if key not in value_d:
                value_d[key] = {}
            value_d = value_d[key]
        if not append:
            value_d[keys[-1]] = value
        else:
            key = keys[-1]
            if key not in value_d:
                value_d[key] = value
            else:
                container = value_d[key]
                assert isinstance(container, list)
                container.append(value)


def format_container_strings(container, **kwargs):
    if isinstance(container, dict):
        for x, y in container.items():
            if isinstance(y, (dict, list)):
                format_container_strings(y, **kwargs)
            elif isinstance(y, str):
                container[x] = y.format(**kwargs)
    elif isinstance(container, list):
        for x, y in enumerate(container):
            if isinstance(y, (dict, list)):
                format_container_strings(y, **kwargs)
            elif isinstance(y, str):
                container[x] = y.format(**kwargs)


def get_last_path(path):
    paths = os.listdir(path)
    paths.sort()
    if not paths:
        raise AssertionError(f"No paths found in {path}")
    return f"{path}/{paths[-1]}"


def get_last_index(path):
    items = os.listdir(path)
    items.sort()
    if not items:
        return 0
    else:
        return int(items[-1])


def mark_path(path, name, uid=None, truncate=4):
    uid = uid or uuid.uuid4().hex
    if truncate:
        uid = uid[:truncate]
    with open(f"{path}/uid.{name}_{uid}", "w") as f:
        f.write(uid)


def find_marks(directory, prefix):
    return [
        "_".join(f.name.split("_")[1:]) for f in Path(directory).iterdir() if f.is_file() and f.name.startswith(prefix)]


def clear_marks(directory, prefix, log=None):
    marks = find_marks(directory, prefix)
    for mark in marks:
        path = f"{directory}/{prefix}_{mark}"
        if log:
            log(f"Removing: {path}")
        os.remove(path)


def get_file_count(directory):
    return sum(1 for _ in Path(directory).rglob('*') if _.is_file())


def cp_markers(src_path, dst_path, prefix="uid."):
    items = os.listdir(src_path)
    for item in items:
        if item.startswith(f"{prefix}"):
            shutil.copyfile(f"{src_path}/{item}", f"{dst_path}/{item}")


def try_read_env_var(env_var_name):
    try:
        return os.environ[env_var_name]
    except KeyError:
        raise AssertionError(f"No env var for key: {env_var_name}")


def run_bash(cmd, cwd="."):
    subprocess.run(["bash", "-c", cmd], cwd=cwd)


def make_dummy(**attrs):
    class Dummy:
        def __getattr__(self, item):
            try:
                return self.__dict__[item]
            except KeyError:
                return None
    dummy = Dummy()
    dummy.__dict__.update(**attrs)
    return dummy


def round_float(float_, dec_figures=2):
    return float(f"{float_:.{dec_figures}f}")


def convert_to_bool(value, log_error=None):
    values_true = ["1", "true", "True", "TRUE", "yes", "Yes", "YES"]
    values_false = ["0", "false", "False", "FALSE", "no", "No" "NO"]
    if value in values_false:
        return False
    elif value in values_true:
        return True
    else:
        if log_error:
            log_error(f"Couldn't convert to boolean: {value} type={type(value)}")
            log_error(f"Allowed for 'True': {values_true}")
            log_error(f"Allowed for 'False': {values_false}")
        raise ValueError("Failed conversion to boolean.")


def bindings_from_args(bindings_kvp, bindings_json):
    type_hint_map = {"int": int, "bool": bool, "float": float, "string": str, "str": str}
    bindings = dict()
    if bindings_json:
        bindings.update(json.loads(bindings_json))
    if bindings_kvp:
        for kvp in bindings_kvp:
            parts = kvp.split(":")
            try:
                key = parts[0]
                value = ":".join(parts[1:-1])
                type_hint = parts[-1]
            except IndexError:
                raise ValueError(
                    f"While trying to split kvp on ':' found {len(parts)} parts, need 3: {kvp}")
            if not type_hint:
                bindings[key] = value
            else:
                try:
                    type_init = type_hint_map[type_hint]
                    if type_init == bool:
                        value = convert_to_bool(value)
                    else:
                        value = type_init(value)
                    bindings[key] = value
                except KeyError:
                    raise ValueError(
                        f"Unknown type hint '{type_hint}' in kvp: {kvp}. Available: {type_hint_map.keys()}")
    return bindings


def get_dir_size(directory_path, humanise=True):
    total_size = 0
    for dir_path, dir_names, filenames in os.walk(directory_path):
        for filename in filenames:
            file_path = os.path.join(dir_path, filename)
            # Skip if it is a broken symbolic link
            if not os.path.islink(file_path):
                total_size += os.path.getsize(file_path)

    # Convert to human-readable format
    if not humanise:
        return total_size
    return humanize.naturalsize(total_size)


def estimate_split_points(dataset_path, seq_split: list) -> int:
    seqs_path = PathHelper(dataset_path).sequences().path()
    seq_names = os.listdir(seqs_path)

    seq_split_size = 0
    for seq_name in seq_names:
        if seq_name not in seq_split:
            continue
        seq_path = PathHelper(seqs_path, seq_name).bins().path()
        seq_split_size += get_dir_size(seq_path, humanise=False)

    return int(seq_split_size / 16)


def count_split_points(dataset_path, seq_split: list) -> int:
    seqs_path = PathHelper(dataset_path).sequences().path()
    seq_names = os.listdir(seqs_path)

    seq_split_points = 0
    for seq_name in seq_names:
        if seq_name not in seq_split:
            continue
        bins_path = PathHelper(seqs_path, seq_name).bins().path()
        bin_names = os.listdir(bins_path)

        for bin_name in bin_names:
            bin_path = f"{bins_path}/{bin_name}"
            arr = np.fromfile(bin_path, dtype=np.float32)
            seq_split_points += len(arr) / 4  # 4D per point.

    return int(seq_split_points)


def count_labels(dataset_path):
    labels_path = f"{dataset_path}/dataset/sequences/.full/labels"

    label_files = os.listdir(labels_path)
    label_files.sort()

    label_counts = {}

    progress = tqdm(desc="Progress", total=len(label_files))
    for label_file in label_files:
        label_path = f"{labels_path}/{label_file}"
        label_data = read_label_data(label_path)
        count_labels_2(label_data, label_counts)
        progress.update(1)

    progress.close()

    return label_counts


def count_labels_2(data, counters: dict):
    keys, counts = np.unique(data, return_counts=True)
    for i, key in enumerate(keys):
        key_int = int(key)
        val_int = int(counts[i])
        counters.setdefault(key_int, 0)
        counters[key] += val_int


def human_readable_to_bytes(size_str):
    import re

    # Match the numeric value and the unit.
    match = re.match(r'([0-9.]+)\s*([KMGTPEZY]?B)', size_str.strip().upper())
    if not match:
        raise ValueError(f"Invalid size format: {size_str}")

    number, unit = match.groups()
    number = float(number)

    # Define unit multipliers.
    units = {
        "B": 1,
        "KB": 10**3,
        "MB": 10**6,
        "GB": 10**9,
        "TB": 10**12,
        "PB": 10**15,
        "EB": 10**18,
        "ZB": 10**21,
        "YB": 10**24,
    }

    if unit not in units:
        raise ValueError(f"Unknown unit: {unit}")

    return int(number * units[unit])


def get_git_sha():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return ""


def get_file_md5(path):
    try:
        return subprocess.check_output(['md5sum', path]).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return ""


def check_process_running(process_name):
    return check_output("pgrep", process_name, converter=int)


def search_for_window(window_name):
    return check_output("xdotool", "search", window_name, converter=int)


def activate_window(window_id):
    return run_bash(f"xdotool windowactivate {window_id}")


def kill_process(process_name):
    return check_output("pkill", process_name)


def check_output(*args, converter=None):
    try:
        output = subprocess.check_output(args).strip()
        if converter:
            output = converter(output)
        return True, output
    except subprocess.CalledProcessError:
        return False, None


def screenshotter(file_log, zoom, up, right):
    process_name = "labeler"
    file_log.log(f"Waiting for instance: {process_name}")

    while True:
        found, pid = check_process_running(process_name)
        if found:
            file_log.log(f"Found instance: ({process_name}, {pid})")
            break
        sleep(1)

    # Waiting window focus.
    sleep(2)

    def xdo(action_type, args, repeat=None, delay=100):
        run_bash(
            f"xdotool {action_type} "
            f"{f'--repeat {repeat}' if repeat else ''} "
            f"{f'--delay {delay}' if delay else ''} "
            f"{args}")

    # Search window and activate.
    ok, window_id = search_for_window("Point Labeler")
    if not ok:
        raise AssertionError("Couldn't find point labeler window!")
    activate_window(window_id)

    # This is i3 default for fullscreen. :)
    xdo("key", "super+f")

    # Waiting for fullscreen.
    sleep(2)

    # First we handle zoom. Mouse button 4/5 (wheel) handle zoom in/out.
    if zoom:
        if zoom < 0:
            file_log.log(f"Zooming out: {-zoom}")
            xdo("click", 5, -zoom)
        else:
            file_log.log(f"Zooming in: {zoom}")
            xdo("click", 4, zoom)

    # Move up/down (w/s).
    if up:
        if up < 0:
            file_log.log(f"Moving down: {-up}")
            xdo("key", "s", -up)
        else:
            file_log.log(f"Moving up: {up}")
            xdo("key", "w", up)

    # Move left/right (a/d).
    if right:
        if right < 0:
            file_log.log(f"Moving left: {-right}")
            xdo("key", "a", -right)
        else:
            file_log.log(f"Moving right: {right}")
            xdo("key", "d", right)

    # Take screenshot (custom point_labeler edit).
    xdo("key", "ctrl+b")

    # Wait for screenshot.
    sleep(1)

    # Kill labeler.
    file_log.log(f"Killing process: {process_name}")
    kill_process(process_name)

    # Wait to die.
    file_log.log("Done screenshotting.")
    file_log.log(f"Killing process: {process_name}")


def read_screenshot_config(screenshot_config_s, screenshot_config_path, log=None):
    parts = screenshot_config_s.split("|")
    if parts[0] == "from-file":
        try:
            with open(screenshot_config_path) as f:
                screenshot_config_s = f.read()
            if log:
                log(f"Read screenshot config from here: {screenshot_config_path}")
        except FileNotFoundError:
            if len(parts) > 1:
                screenshot_config_s = parts[1]
            else:
                # Default default.
                screenshot_config_s = ":-10:0:0"
    parts = screenshot_config_s.split(":")
    return int(parts[1]), int(parts[2]), int(parts[3])


def load_config(config_cls, bindings_path, config_path):
    bindings = read_yaml(bindings_path)
    config_loader = JinjaYamlLoader(config_path, lambda **x: config_cls(x))
    config = config_loader.load(**bindings)
    num_points = int(config.model["num_points"])
    num_layers = int(config.model["num_layers"])
    num_neighbors = int(config.model["num_neighbors"])

    # For lower points we need to reduce layers.
    if num_points < 512:
        num_layers = min(num_layers, 3)
    if num_points < 128:
        num_layers = min(num_layers, 2)
    if num_points < 32:
        num_layers = min(num_layers, 1)
    if num_points < 8:
        num_layers = min(num_layers, 0)
    if num_points < 2:
        raise ValueError("This won't work. Use points >= 2, please.")

    # Type preservation via jinja2 doesn't quite work here.
    config.model["num_points"] = int(num_points)
    config.model["num_layers"] = int(num_layers)
    config.model["num_neighbors"] = int(num_neighbors)
    return config


def write_yaml(path, data):
    with open(path, "w") as f:
        YAML().dump(data, f)
