from abc import ABC, abstractmethod


class _Enum:
    @classmethod
    def choices(cls):
        return [y for x, y in cls.__dict__.items() if not x.startswith("_") and not isinstance(y, classmethod)]


class _Parser(ABC):
    @classmethod
    @abstractmethod
    def add_args(cls, parent_parser):
        raise NotImplementedError()


class _Args(ABC):
    pass
