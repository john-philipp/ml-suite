import os

from src.file_helpers import rm
from src.file_logger import FileLogger
from src.path_handler import PathHandler
from src.this_env import GLOBALS


def handle_clean(root_path, all_, index, log=GLOBALS.log):
    file_log = FileLogger()

    if all_:
        file_log.log(f"Removing: {file_log.link(root_path)}")
        if not os.path.isdir(root_path):
            file_log.log("Nothing to remove.")
        else:
            rm(root_path, log_progress=True)

    elif index:
        path_handler = PathHandler(root_path)
        try:
            index_path = path_handler.get_path(index)
        except AssertionError:
            file_log.log("Nothing to remove.")
        else:
            file_log.log(f"Removing: {file_log.link(index_path)}")
            rm(index_path, log_progress=True)
