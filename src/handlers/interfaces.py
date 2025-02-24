from abc import ABC, abstractmethod


class _Handler(ABC):

    Args = None

    def __init__(self, args):
        self.args = self.arg_cls()(args)

    @classmethod
    def arg_cls(cls):
        return cls.Args

    @abstractmethod
    def handle(self, *_):
        raise NotImplementedError()
