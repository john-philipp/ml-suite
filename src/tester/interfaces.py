from abc import ABC, abstractmethod


class ITester(ABC):
    @abstractmethod
    def test(self):
        raise NotImplementedError()
