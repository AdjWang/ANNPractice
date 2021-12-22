""" ReLU activation function """
from .basic import NNFunction
from matrix import Matrix
from math import pow, e

class Sigmoid(NNFunction):
    def __init__(self, input_channel: int):
        self.input_channel = input_channel

    def sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + pow(e, -x))

    def diff_sigmoid(self, x: float) -> float:
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))

    def forward(self, x: Matrix) -> Matrix:
        assert x.shape == (self.input_channel, 1)
        self.x = x
        return x.apply(self.sigmoid)

    def backward(self, y: Matrix) -> Matrix:
        assert y.shape == (self.input_channel, 1)
        return self.x.apply(self.diff_sigmoid) * y
