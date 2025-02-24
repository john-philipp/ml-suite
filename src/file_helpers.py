import os
import shutil
import threading
import time
from pathlib import Path

from tqdm import tqdm

from src.methods import run_bash
from src.this_env import GLOBALS


_log = GLOBALS.log


def pj(*args, **kwargs) -> str:
    return os.path.join(*args, **kwargs).__str__()


def rm(path, log_progress=False, log_sleep=1, **shutil_kwargs):
    if not os.path.isdir(path):
        return

    logging_thread = None

    if log_progress:
        logging_thread = threading.Thread(target=progress_logger_rm, args=(path, log_sleep))
        logging_thread.start()

    shutil.rmtree(path, **shutil_kwargs)

    if logging_thread:
        logging_thread.join()


def mv(src, dst, log_progress=False, log_sleep=1, **shutil_kwargs):
    logging_thread = None

    if log_progress:
        logging_thread = threading.Thread(target=progress_logger, args=(src, dst, log_sleep))
        logging_thread.start()

    shutil.move(src, dst, **shutil_kwargs)

    if logging_thread:
        logging_thread.join()


def cp(src, dst, log_progress=False, log_sleep=1, rsync=False, rsync_args="", **shutil_kwargs):
    logging_thread = None

    if log_progress:
        logging_thread = threading.Thread(target=progress_logger, args=(src, dst, log_sleep))
        logging_thread.start()

    if rsync:
        run_bash(f"rsync -a --delete {src}/* {dst} {rsync_args}", ".")
    else:
        shutil.copytree(src, dst, **shutil_kwargs)

    if log_progress:
        logging_thread.join()


def mk(path, exist_ok=True, log=False):
    os.makedirs(path, exist_ok=exist_ok)


def get_file_count(directory):
    try:
        return sum(1 for _ in Path(directory).rglob('*') if _.is_file())
    except FileNotFoundError:
        return 0


def progress_logger(src, dst, log_sleep):
    file_count = get_file_count(src)
    progress = tqdm(desc="Progress", total=file_count)

    while True:
        current_count = get_file_count(dst)
        progress.n = current_count
        progress.update(0)
        if current_count >= file_count:
            progress.close()
            break
        time.sleep(log_sleep)


def progress_logger_rm(rm_dir, log_sleep):
    file_count = get_file_count(rm_dir)
    progress = tqdm(desc="Progress", total=file_count)

    while True:
        current_count = get_file_count(rm_dir)
        progress.n = file_count - current_count
        progress.update(0)
        if current_count <= 0:
            progress.close()
            break
        time.sleep(log_sleep)
