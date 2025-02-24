import os
import sys

from src.parsers.args import parse_input_args
from src.parsers.enums import ModeType, ScriptsActionType
from src.file_helpers import mk
from src.file_logger import FileLogger
from src.handlers import get_action_handler
from src.handlers.scripts.handler_scripts_run import HandlerScriptsRun
from src.path_helper import PathHelper, PathHelperConfig
from src.paths import PATHS
from src.this_env import GLOBALS


log = GLOBALS.log
file_log = FileLogger()


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        args.append("-h")

    file_log.log(80 * "-")
    file_log.log(f"Called with args: {sys.argv[1:]}")

    args = parse_input_args(*args)


    def main(*args_, parsed_args=None):

        ph_config = PathHelperConfig(auto_rollback=True)
        ph = PathHelper(config=ph_config).generated().commit()
        mk(ph.logs().path())
        mk(ph.recordings().path())
        mk(ph.datasets().path())
        mk(ph.trainings().path())
        mk(ph.predictions().path())
        mk(ph.screenshots().path())
        mk(ph.weights().path())
        mk(PATHS.snapshots)
        mk(PATHS.results)

        if not os.path.exists(".results"):
            os.symlink(PATHS.results, ".results")

        if not os.path.exists(".snapshots"):
            os.symlink(PATHS.snapshots, ".snapshots")

        if not parsed_args:
            parsed_args = parse_input_args(*args_)

        if parsed_args.mode == ModeType.SCRIPTS:
            if parsed_args.action == ScriptsActionType.RUN:
                file_log.log("Running scripts...")
                HandlerScriptsRun(parsed_args).handle(main)
                file_log.log("Scripting done.")
        else:
            action_handler = get_action_handler(parsed_args.mode, parsed_args.action)
            action_handler(parsed_args).handle()

    main(parsed_args=args)

    file_log.close()
