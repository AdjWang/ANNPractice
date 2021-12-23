""" ReLU activation function """
from .basic import NNFunction
from matrix import Matrix


class ReLU(NNFunction):
    def __init__(self, input_channel: int):
        self.input_channel = input_channel
        self.gradient = 1.0

    def relu(self, x: float) -> float:
        return x * self.gradient if x > 0.0 else 0.0

    def diff_relu(self, x: float) -> float:
        return self.gradient if x > 0.0 else 0.0

    def forward(self, x: Matrix) -> Matrix:
        assert x.shape == (self.input_channel, 1)
        self.x = x
        return x.apply(self.relu)

    def backward(self, y: Matrix) -> Matrix:
        assert y.shape == (self.input_channel, 1)
        return self.x.apply(self.diff_relu) * y
