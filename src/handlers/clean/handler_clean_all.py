from src.parsers.interfaces import _Args
from src.file_helpers import rm
from src.handlers.interfaces import _Handler


class HandlerCleanAll(_Handler):

    class Args(_Args):
        def __init__(self, args):
            pass

    def handle(self):
        args: HandlerCleanAll.Args = self.args
        rm("_generated", log_progress=True)
