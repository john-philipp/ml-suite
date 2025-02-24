import argparse
from argparse import _StoreTrueAction

from src.parsers.args import DESCRIPTION
from src.parsers.interfaces import _Args
from src.file_logger import FileLogger
from src.handlers.interfaces import _Handler
from src.parsers.parser_main import ParserMain
from src.this_env import GLOBALS


class HandlerMiscMakeDocs(_Handler):

    class Args(_Args):
        def __init__(self, args):
            pass

    def handle(self):
        file_log = FileLogger()
        file_log.log("Making docs...")

        arg_parser = argparse.ArgumentParser(
            prog=GLOBALS.app_name,
            description=DESCRIPTION)

        ParserMain.add_args(arg_parser)
        print(arg_parser.description)

        # To avoid duplicates on screen.
        skip = set()

        pad1 = 20
        pad2 = 16
        pad3 = 8

        actions = arg_parser._actions[1]
        choices = actions.choices
        for name, sub_parser in choices.items():
            if sub_parser.prog in skip:
                continue
            skip.add(sub_parser.prog)

            print(80 * "=")
            print(f"{sub_parser.prog:{pad1}}: {sub_parser.description}")
            print(80 * "=")

            sub_parsers_2 = sub_parser._actions[1]
            choices_2 = sub_parsers_2.choices
            first = True
            for name_2, sub_parser_2 in choices_2.items():

                if sub_parser_2.prog in skip:
                    continue
                skip.add(sub_parser_2.prog)

                if first:
                    print(f" .")
                    print(f"  +", end="")
                    first = False
                else:
                    print(f"  +", end="")

                print(f" {sub_parser_2.prog.split(' ')[-1]:{pad2}}: {sub_parser_2.description}")

                for i, action in enumerate(sub_parser_2._actions[1:]):
                    help_str = ""
                    if action.help is not None:
                        help_str = action.help.replace("%(default)s", f"{action.default}")

                    if action.required:
                        help_str = f"Required. {help_str}"

                    type_ = "str"
                    if action.type is not None:
                        type_ = action.type.__name__

                    if action.nargs == "+":
                        type_ += "[]"

                    if isinstance(action, _StoreTrueAction):
                        type_ = None

                    s = ""
                    for x in action.option_strings:
                        if s:
                            s += ", "
                        s += x
                    print(f"   .")
                    print(f"   *  {s}")
                    print(f"   .      {'Help':{pad3}}: {help_str}")
                    if type_:
                        print(f"   .      {'Type':{pad3}}: {type_}")
                    if action.choices:
                        print(f"   .      {'Choices':{pad3}}: {action.choices}")

                print("   ")

        file_log.close()
