from src.parsers.interfaces import _Args
from src.handlers.clean.methods import handle_clean
from src.handlers.interfaces import _Handler
from src.this_env import GLOBALS


log = GLOBALS.log


class HandlerCleanDatasets(_Handler):

    PATH = "_generated/02-datasets"

    class Args(_Args):
        def __init__(self, args):
            self.index = args.index
            self.all = args.all

    def handle(self):
        args: HandlerCleanDatasets.Args = self.args
        handle_clean("_generated/02-datasets", args.all, args.index)
