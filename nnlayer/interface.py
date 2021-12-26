from __future__ import annotations
from typing import List
import pickle

from .sigmoid import Sigmoid
from .basic import NNFunction
from .fc import FullConnection
from matrix import Matrix


class Model(NNFunction):
    @staticmethod
    def new_model(input_channel: int, output_channel: int, hidden: List[int], learning_rate: float) -> Model:
        assert len(hidden) > 0
        if len(hidden) == 1:
            assert hidden[0] == input_channel
            return Model([
                FullConnection(input_channel, output_channel, learning_rate),
                Sigmoid(output_channel)
            ])
        # len(hidden) >= 2 from here
        layers = [
            FullConnection(input_channel, hidden[0], learning_rate),
            Sigmoid(hidden[0])
        ]
        for n in hidden[1:-1]:
            layers.append(FullConnection(n, n, learning_rate))
            layers.append(Sigmoid(n))
        layers.extend([
            FullConnection(hidden[-1], output_channel, learning_rate),
            Sigmoid(output_channel)
        ])
        return Model(layers)

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

    @staticmethod
    def load(filepath: str) -> Model:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model

    def dump(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
