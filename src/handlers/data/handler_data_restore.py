import os
import shutil
from pathlib import Path

from src.parsers.interfaces import _Args
from src.file_helpers import rm, pj, cp, mk
from src.handlers.interfaces import _Handler
from src.methods import get_last_path, mark_path, clear_marks
from src.paths import PATHS
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerDataRestore(_Handler):

    class Args(_Args):
        def __init__(self, args, paths=PATHS):
            self.generated_path = paths.generated
            self.save_path = paths.snapshots
            self.merge = args.merge
            self.rsync = True
            self.id = args.id

    def handle(self):
        args: HandlerDataRestore.Args = self.args

        id_ = args.id
        if id_ == -1:
            latest = get_last_path(args.save_path)
            id_ = latest.split("/")[-1]

        log.info(f"Will restore: {id_}")
        if id_:

            clear_marks(".", "uid.snapshot", log=log.info)

            if not args.merge:
                restore_src = pj(args.save_path, id_)
                restore_dst = args.generated_path
                if not args.rsync:
                    log.info(f" Removing: {restore_dst}")
                    rm(restore_dst, log_progress=True, log_sleep=0.1)
                log.info(f" Restoring: {restore_src} -> {restore_dst}")
                cp(restore_src, restore_dst, log_progress=True, rsync=args.rsync)

            else:
                merge_paths = args.merge.split(",")
                for merge_path in merge_paths:
                    log.info(f" Merging: {merge_path}")
                    merge_src = pj(args.save_path, id_, merge_path)
                    merge_dst = pj(args.generated_path, merge_path)
                    mk(merge_dst)

                    merge_items = os.listdir(merge_src)
                    merge_items.sort()

                    for item in merge_items:
                        log.info(f"   Merging: {merge_path}/{item}")
                        src_path = pj(merge_src, item)
                        dst_path = pj(merge_dst, item)
                        if os.path.isdir(src_path):
                            cp(src_path, dst_path, log_progress=True, dirs_exist_ok=True)
                            # shutil.copytree(src_path, dst_path, )
                        else:
                            shutil.copy2(src_path, dst_path)

            mark_path(".", "snapshot", uid=id_, truncate=None)

    @staticmethod
    def get_file_count(directory):
        return sum(1 for _ in Path(directory).rglob('*') if _.is_file())
