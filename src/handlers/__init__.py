import importlib

from src.handlers.interfaces import _Handler
from src.parsers.enums import ModeType, ActionType


def get_action_handler(mode_type: ModeType, action_type: ActionType) -> _Handler.__class__:

    import_path = _to_import_path(mode_type, action_type)
    handler_name = _to_handler_name(mode_type, action_type)

    print(f">>> {import_path}.{handler_name}")
    mode_module = importlib.import_module(import_path)
    return getattr(mode_module, handler_name)


def _to_import_path(mode_type: ModeType, action_type: ActionType):
    mode_type_s = mode_type.__str__()
    action_type_s = action_type.__str__().replace('-', "_")
    action_type_s = _apply_aliases(action_type_s)
    path = f"src.handlers.{mode_type_s}.handler_{mode_type_s}_{action_type_s}"
    print(path)
    return path


def _to_handler_name(mode_type: ModeType, action_type: ActionType):
    mode_type_camel_s = ""
    for x in _apply_aliases(mode_type.__str__()).split("-"):
        mode_type_camel_s += x[0].upper() + x[1:]
    action_type_camel_s = ""
    for x in _apply_aliases(action_type.__str__()).split("-"):
        action_type_camel_s += x[0].upper() + x[1:]
    path = f"Handler{mode_type_camel_s}{action_type_camel_s}"
    print(path)
    return path


def _apply_aliases(string):
    aliases = {
        "ls": "list",
        "store": "save",
        "load": "restore"
    }
    for from_, to_ in aliases.items():
        if string == from_:
            return to_
        elif string.endswith(f"-{from_}"):
            return string.endswith(f"-{to_}")
    return string
