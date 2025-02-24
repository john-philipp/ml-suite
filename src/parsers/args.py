import argparse
import os

from src.parsers.parser_main import ParserMain
from src.paths import PATHS
from src.this_env import GLOBALS


DESCRIPTION = \
    (f"Handle semantic segmentation data recording and conversion. Data can be saved to and restored "
     f"from {PATHS.snapshots.replace(os.environ['HOME'], '~')} by use of an alias.")


def parse_input_args(*args):

    description = DESCRIPTION

    arg_parser = argparse.ArgumentParser(
        prog=GLOBALS.app_name,
        description=description)

    ParserMain.add_args(arg_parser)
    parsed_args = arg_parser.parse_args(args=args)
    return parsed_args


