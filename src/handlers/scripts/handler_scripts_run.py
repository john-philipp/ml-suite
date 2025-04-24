import os
import sys

from src.parsers.interfaces import _Args
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler
from src.jinja_yaml_loader import JinjaYamlLoader
from src.methods import bindings_from_args, mark_path
from src.this_env import GLOBALS


class HandlerScriptsRun(_Handler):

    class Args(_Args):
        def __init__(self, args):
            self.scripts = args.scripts
            self.bindings_kvp = args.bindings_kvp
            self.bindings_json = args.bindings_json
            self.range = args.range
            self.from_stdin = args.from_stdin

    class Script:
        def __init__(self, actions, **_):
            self.actions = actions

    def handle(self, main):

        args: HandlerScriptsRun.Args = self.args
        bindings = bindings_from_args(args.bindings_kvp, args.bindings_json)

        file_log = FileLogger()
        file_log.log(f"Handler: {self.__class__.__name__}")
        file_log.log(f"Scripts: {args.scripts}")
        file_log.log(f"Bindings: {bindings}")

        class Script:
            def __init__(self, file_path=False, raw_content=False, data=None):
                self.file_path = file_path
                self.raw_content = raw_content
                self.data = data

        scripts = []
        if args.from_stdin:
            scripts.append(Script(raw_content=True, data=sys.stdin.read()))
        else:
            for script_path in args.scripts:
                scripts.append(Script(file_path=True, data=script_path))

        for script in scripts:
            script_path = script.data
            yaml_string = None
            if script.raw_content:
                script_path = "stdin"
                yaml_string = script.data
            if GLOBALS.nested_level == 0:
                if os.path.isdir(script_path):
                    script_path += "/_run.yml"
                mark_path(".", "script", uid="_".join(script_path.split("/")[1:]), truncate=None)

            script = JinjaYamlLoader(script_path, HandlerScriptsRun.Script).load(yaml_string=yaml_string, **bindings)

            start, end = self._range_from_args(args)

            for action_args in script.actions[start:end]:
                # We might have to move to array style notation if
                # splitting by space becomes too limiting. More
                # of a convenience edit for now.
                file_log.log(80 * "-")
                file_log.log(f"Running script action (nested={GLOBALS.nested_level}): {action_args}")
                file_log.log(80 * "-")

                GLOBALS.nest()
                main(*action_args.split(" "))
                GLOBALS.denest()

                file_log.log(f"Done running script action: {action_args}")

        file_log.close()

    @staticmethod
    def _range_from_args(args: Args):
        if not args.range:
            return None, None
        range_s = args.range
        parts = range_s.split(":")
        if len(parts) == 1:
            start = int(parts[0])
            end = None
        elif len(parts) == 2:
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else None
        else:
            raise ValueError(f"Unexpected. Need range of form start:end, found: {range_s}")

        return start, end
