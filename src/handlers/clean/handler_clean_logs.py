from src.parsers.interfaces import _Args
from src.handlers.clean.methods import handle_clean
from src.handlers.interfaces import _Handler


class HandlerCleanLogs(_Handler):

    class Args(_Args):
        def __init__(self, args):
            pass

    def handle(self):
        handle_clean(f"_generated/00-logs", True, -1)
