import os

from src.handlers.data.handler_data_annotate import HandlerDataAnnotate
from src.parsers.interfaces import _Args
from src.file_helpers import pj, cp
from src.handlers.interfaces import _Handler
from src.methods import get_timestamp, _read_metadata, set_data, _write_metadata, make_dummy, mark_path, find_marks, \
    clear_marks
from src.paths import PATHS
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerDataSave(_Handler):

    class Args(_Args):
        def __init__(self, args, paths=PATHS):
            self.working_dir = paths.generated
            self.save_dir = paths.snapshots

            self.annotations = args.annotations
            self.save_alias = args.alias
            self.into_id = args.into_id
            self.rsync = True

    def handle(self):
        args: HandlerDataSave.Args = self.args

        if args.annotations:
            input_args = make_dummy(add_annotations=args.annotations)
            handler_args = HandlerDataAnnotate.Args(input_args)
            handler = HandlerDataAnnotate(handler_args)
            handler.handle()

        os.makedirs(args.working_dir, exist_ok=True)
        save_dir_src = args.working_dir

        id_ = args.into_id
        if not id_:
            id_ = get_timestamp()
        save_dir_dst = pj(args.save_dir, id_)

        metadata_d = _read_metadata(args.working_dir)
        set_data(metadata_d, args.save_alias, "annotations.alias")
        _write_metadata(args.working_dir, metadata_d)

        cp(save_dir_src, save_dir_dst, log_progress=True, rsync=args.rsync)
        log.info(f" Saved {save_dir_src} to {save_dir_dst}")

        clear_marks(".", "uid.snapshot")
        mark_path(".", "snapshot", uid=id_, truncate=None)

        return id_
