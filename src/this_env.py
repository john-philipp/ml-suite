import logging

APP_NAME = "mls"
_log = logging.getLogger(APP_NAME)
logging.basicConfig(level=logging.INFO)


class Globals:
    nested_level = 0
    app_name = APP_NAME
    script_dir = "."
    log = _log

    def nest(self):
        self.nested_level += 1

    def denest(self):
        if self.nested_level != 0:
            self.nested_level -= 1

    def set_script_dir(self, script_dir):
        self.script_dir = script_dir


GLOBALS = Globals()
