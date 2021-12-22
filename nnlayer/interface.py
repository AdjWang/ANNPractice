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
        return y


class Output(NNFunction):
    def __init__(self, input_channel: int):
        self.input_channel = input_channel

    def forward(self, x: Matrix) -> Matrix:
        assert x.shape == (self.input_channel, 1)
        # return softmax(x)
        return x

    def backward(self, y: Matrix) -> Matrix:
        """
        Args:
            yt: ground truth
        """
        assert y.shape == (self.input_channel, 1)
        return y
