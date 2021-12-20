from .basic import NNFunction
from .algorithms import softmax
from matrix import Matrix


class Input(NNFunction):
    def __init__(self, input_channel: int):
        self.input_channel = input_channel

    def forward(self, x):
        assert x.shape == (self.input_channel, 1)
        return x

    def backward(self, y):
        pass


class Output(NNFunction):
    def __init__(self, input_channel: int):
        self.input_channel = input_channel

    def forward(self, x: Matrix) -> Matrix:
        assert x.shape == (self.input_channel, 1)
        return softmax(x)

    def backward(self, y: Matrix) -> Matrix:
        assert y.shape[1] == self.input_channel
        return y
