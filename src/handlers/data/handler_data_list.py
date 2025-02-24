import os
import re

import humanize

from src.parsers.interfaces import _Args
from src.file_helpers import pj
from src.handlers.interfaces import _Handler
from src.methods import _read_metadata
from src.paths import PATHS


class HandlerDataList(_Handler):

    class Args(_Args):
        def __init__(self, args, paths=PATHS):
            self.annotations_filter_re = args.annotations_filter_re
            self.id_filter_re = args.id_filter_re

            self.save_path = paths.snapshots
            self.size = args.size

    def handle(self):
        args: HandlerDataList.Args = self.args

        filter_res = args.annotations_filter_re
        if filter_res:
            print()
            print("Filters:")
            print(f"  ID:    {args.id_filter_re or '.*'}")
            print("  Annotations:")
            for filter_re in filter_res:
                print(f"    {filter_re}")

        available_ids = os.listdir(args.save_path)
        available_ids.sort()

        print()
        print("Recording/dataset(s) found:")
        filtered_ids = {}
        exclude = set()

        for pattern in args.id_filter_re or []:
            for id_ in available_ids:
                if not re.match(pattern, id_):
                    exclude.add(id_)

        for id_ in available_ids:
            if id_ in exclude:
                continue
            metadata = _read_metadata(pj(args.save_path, id_))
            try:
                annotations = metadata["annotations"]
            except KeyError:
                annotations = {}

            if not filter_res:
                filtered_ids[id_] = annotations
                continue

            for filter_re in filter_res:
                key, pattern = filter_re.split("=")
                ok = False
                if key not in annotations and not pattern:
                    filtered_ids[id_] = annotations
                    ok = True
                elif key in annotations:
                    if re.match(pattern, annotations[key]):
                        filtered_ids[id_] = annotations
                        ok = True
                if not ok:
                    exclude.add(id_)
                    break

        filtered_keys = list(filtered_ids.keys())
        filtered_keys.sort()

        for id_ in filtered_keys:

            if id_ in exclude:
                continue

            size = self._get_dir_size(pj(args.save_path, id_)) if args.size else -1
            annotations = filtered_ids[id_]
            annotations_list = [f"{x}={y}" for x, y in annotations.items()]
            annotations_list.sort()
            annotations_s = ""
            for annotation in annotations_list:
                annotations_s += f"{annotation} "
            print(f"  id: {id_} \tsize: {size} \tannotations: {annotations_s}")

    @staticmethod
    def _get_dir_size(directory_path):
        total_size = 0
        for dir_path, dir_names, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dir_path, filename)
                # Skip if it is a broken symbolic link
                if not os.path.islink(file_path):
                    total_size += os.path.getsize(file_path)

        # Convert to human-readable format
        return humanize.naturalsize(total_size)


