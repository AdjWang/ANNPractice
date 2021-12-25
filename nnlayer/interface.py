from typing import List

from .basic import NNFunction
from matrix import Matrix


class Model(NNFunction):
    def __init__(self, layers: List[NNFunction]):
        self.layers = layers

    def forward(self, x: Matrix) -> Matrix:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y: Matrix) -> Matrix:
        for layer in reversed(self.layers):
            y = layer.backward(y)
        return y

