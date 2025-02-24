import os
import shutil
import time
from datetime import datetime
from pathlib import Path

from ruamel.yaml import YAML
import yaml

from src.methods import set_data
from src.this_env import GLOBALS


def get_timestamp(iso=True):
    now = datetime.now()
    if iso:
        return now.isoformat()
    return (now.__str__()
            .replace(" ", "_")
            .replace(":", "")
            .replace("-", "")
            .replace(".", "-")[:-3])


class FileLogger:

    MAIN_LOG = "_generated/00-logs/log.txt"

    def __init__(self, path=None):
        os.makedirs(os.path.dirname(FileLogger.MAIN_LOG), exist_ok=True)

        self._p_file = open(FileLogger.MAIN_LOG, "a")
        self._start_time = time.time()
        self._root_path = path

        self._root_info_file_path = None
        self._s_info_file_path = None
        self._s_log_dir_path = None
        self._s_file_path = None
        self._s_file = None
        self.timestamp = get_timestamp(iso=False)

        if path:
            log_path = f"{path}/log"
            log_path_2 = f"{log_path}/{self.timestamp}"
            path = f"{log_path_2}/log.txt"

            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)

            self._s_log_dir_path = log_path_2
            self._s_file_path = path
            self._s_file = open(path, "a")

            info_path = path.replace("log.txt", "info.yml")
            Path(info_path).touch()

            self._s_info_file_path = info_path
            self._root_info_file_path = f"{self._root_path}/{os.path.basename(self._s_info_file_path)}"

            prev_log_dirs = os.listdir(log_path)
            prev_log_dirs.sort()
            prev_log_dirs = prev_log_dirs[:-1]

            # We base info.yml on previous invocation.
            # Look for last non-empty info.yml.
            for prev_log_dir in prev_log_dirs[::-1]:
                prev_info_yml = f"{log_path}/{prev_log_dir}/info.yml"
                if os.path.isfile(prev_info_yml):
                    if os.path.getsize(prev_info_yml) > 0:
                        shutil.copyfile(prev_info_yml, info_path)
                        self.log(f"Basing on previous info.yml: {self.link(prev_info_yml)}")
                        break

            self.log(f"Logs written here: {self.link(self.get_log_dir())}")

    def get_log_dir(self):
        return self._s_log_dir_path

    @staticmethod
    def link(path):
        return f"file://{os.path.abspath(path)}"

    def log(self, msg, file=None, logger=None):

        if not logger:
            logger = GLOBALS.log.info
        logger(msg)

        if file:
            file.write(msg)
            file.flush()

        msg = f"{get_timestamp(iso=True)}: {msg}\n"

        self._p_file.write(msg)
        self._p_file.flush()

        if self._s_file:
            self._s_file.write(msg)
            self._s_file.flush()

    def log_and_raise(self, msg, ex_cls=AssertionError):
        self.log(msg)
        self.close()
        raise ex_cls(msg)

    def keep_file(self, path):
        if self._s_log_dir_path:
            file_name = os.path.basename(path)
            shutil.copyfile(path, f"{self._s_log_dir_path}/{file_name}")

    def add_infos(self, local=False, root=False, append=False, use_ruamel=False, **infos):
        if self._s_info_file_path and local:
            self._add_infos(self._s_info_file_path, append, use_ruamel, **infos)
        if self._root_info_file_path and root:
            self._add_infos(self._root_info_file_path, append, use_ruamel, **infos)

    @staticmethod
    def _add_infos(info_path, append, use_ruamel=False, **infos):
        load = yaml.safe_load
        dump = yaml.safe_dump

        if use_ruamel:
            yml = YAML()
            yml.default_flow_style = True
            load = yml.load
            dump = yml.dump

        if os.path.isfile(info_path):
            with open(info_path, "r") as f:
                data = load(f) or {}
        else:
            data = {}
        for key_path, value in infos.items():
            set_data(data, value, key_path, append=append)
        with open(info_path, "w") as f:
            dump(data, f)

    def close(self):
        execution_time_s = time.time() - self._start_time
        execution_time_s = float(f"{execution_time_s:.2f}")
        self.log(f"Completed in {execution_time_s:.2f}s.")
        self._p_file.close()

        if self._s_file:
            self._s_file.close()

        if self._s_file_path:

            # We append to the root log.
            root_log = f"{self._root_path}/{os.path.basename(self._s_file_path)}"
            Path(root_log).touch()
            with open(self._s_file_path, "r") as f:
                lines = f.readlines()
            with open(root_log, "a") as f:
                f.write(f"{80 * '-'}\n")
                f.writelines(lines)
