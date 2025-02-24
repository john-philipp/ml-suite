from abc import ABC, abstractmethod


class ITrainer(ABC):
    @abstractmethod
    def train(self):
        raise NotImplementedError()
