from src.parsers.enums import ScriptsActionType
from src.parsers.interfaces import _Parser


class ParserScriptsRun(_Parser):
    @classmethod
    def add_args(cls, parent_parser):

        parser = parent_parser.add_parser(
            description="Run some scripts. Scripts allow deterministic definition "
                        "of complex training plans. Scripts may nest.",
            name=ScriptsActionType.RUN,
            help="Run scripts.")

        arg_group = parser.add_mutually_exclusive_group(required=True)

        arg_group.add_argument(
            "--scripts", "-s",
            help="Run script(s). Allows chaining.",
            nargs="+")

        arg_group.add_argument(
            "--from-stdin", "-i",
            help="Read script from stdin.",
            action="store_true")

        parser.add_argument(
            "--local-paths", "-l",
            help="Nested script references are local.",
            action="store_true")

        parser.add_argument(
            "--range", "-r",
            help="Only execute steps in range 'start:end' (starts at 0) (default=%(default)s).",
            default="0:")

        parser.add_argument(
            "--bindings-kvp", "-b",
            help="Define bindings as kvp pairs (key:value[:int|float|str|bool]). Takes priority over JSON.",
            nargs="+")

        parser.add_argument(
            "--bindings-json", "-b2",
            help="Define bindings as a JSON string for more complex bindings.")

        return parser
