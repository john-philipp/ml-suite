from src.parsers.interfaces import _Args
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler


class HandlerMiscEcho(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.msgs = args.msgs

    def handle(self):
        args: HandlerMiscEcho.Args = self.args
        file_log = FileLogger()
        file_log.log(" ".join(args.msgs))
        file_log.close()
