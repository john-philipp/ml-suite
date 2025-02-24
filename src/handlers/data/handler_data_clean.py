import os
import secrets

from src.parsers.interfaces import _Args
from src.file_helpers import rm, pj, mv
from src.handlers.interfaces import _Handler
from src.paths import PATHS
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerDataClean(_Handler):

    class Args(_Args):
        def __init__(self, args, paths=PATHS):
            self.clean_generated = args.generated
            self.clean_training = args.paths_training
            self.no_backup = args.no_backup
            self.clean_aliases = args.ids
            self.clean_kitti = args.kitti
            self.clean_all = args.all
            self.paths = paths

    def handle(self):
        args: HandlerDataClean.Args = self.args

        if args.clean_all:
            rm(args.paths.generated)
            saved_aliases = os.listdir(args.paths.snapshots)
            saved_aliases.sort()

            for saved_alias in saved_aliases:
                self._handle_sanity_save(pj(args.paths.snapshots, saved_alias), args.paths.save_sanity)
            rm(args.paths.snapshots)
        else:
            if args.clean_generated:
                rm(args.paths.generated)
            if args.clean_kitti:
                rm(args.paths.kitti)
            if args.clean_training:
                rm(args.paths.paths_training)
            if args.clean_aliases:
                for clean_alias in args.clean_aliases:
                    save_path = pj(args.paths.snapshots, clean_alias)
                    if not args.no_backup:
                        self._handle_sanity_save(save_path, args.paths.save_sanity)
                    else:
                        rm(save_path)
                    if os.path.isdir(save_path):
                        raise ValueError(f"Directory {save_path} still exists.")

    def _handle_sanity_save(self, save_dir_dst, save_sanity_dir):
        os.makedirs(save_sanity_dir, exist_ok=True)
        sanity_backup_path = self._find_missing_path(save_sanity_dir)
        log.info(f" Sanity backup {save_dir_dst} -> {sanity_backup_path}")
        if sanity_backup_path.startswith("/tmp"):
            log.debug("Note, '/tmp' will be cleared on OS restart.")
        try:
            mv(save_dir_dst, sanity_backup_path, log_progress=True)
        except FileNotFoundError as ex:
            log.warn(f"Not found: {ex}")

    @staticmethod
    def _find_missing_path(base):
        path = ""
        while not path or os.path.isdir(path):
            path = pj(base, secrets.token_hex(6))
        return path
