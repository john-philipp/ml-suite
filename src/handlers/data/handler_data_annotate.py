import json
import os

from src.parsers.interfaces import _Args
from src.file_helpers import pj
from src.handlers.interfaces import _Handler
from src.paths import PATHS
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerDataAnnotate(_Handler):
    
    class Args(_Args):
        def __init__(self, args):
            self.add_annotations = args.add_annotations
            self.rm_annotations = args.rm_annotations
            self.id = args.id

    def handle(self):
        args: HandlerDataAnnotate.Args = self.args

        directory = PATHS.generated
        save_path = PATHS.snapshots
        if args.id:
            directory = pj(save_path, args.id)
            if not os.path.isdir(directory):
                raise ValueError(f"Couldn't find directory: {directory}")

        metadata_path = pj(directory, "metadata.yml")
        if not os.path.isfile(metadata_path):
            with open(metadata_path, "w") as f:
                f.write("{}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        metadata_annotations = metadata.setdefault("annotations", {})

        log.info(f" Annotating {os.path.basename(directory)}:")
        if args.add_annotations:
            for annotation_kvp in args.add_annotations:
                # Value can't contain spaces.
                # print(annotation_kvp)
                key, value = annotation_kvp.split("=")
                metadata_annotations[key] = value
                log.info(f"  + {key}={value}")

        if args.rm_annotations:
            for key in args.rm_annotations:
                if key in metadata_annotations:
                    value = metadata_annotations[key]
                    del metadata_annotations[key]
                    log.info(f"  - {key}={value}")

        with open(metadata_path, "w") as f:
            f.write(json.dumps(metadata, indent=2))
