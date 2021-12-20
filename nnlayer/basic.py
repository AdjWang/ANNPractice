from abc import ABC, abstractmethod


class NNFunction(ABC):
    """ Abstract class of nnlayer. """

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, y):
        pass
