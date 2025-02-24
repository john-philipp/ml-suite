from src.parsers.interfaces import _Args
from src.file_helpers import rm
from src.handlers.interfaces import _Handler


class HandlerCleanRecordings(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.index = args.index
            self.all = args.all

    def handle(self):
        args: HandlerCleanRecordings.Args = self.args
        rm(f"_generated/04-predictions")
