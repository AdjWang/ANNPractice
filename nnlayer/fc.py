""" full connection layer """
from .basic import NNFunction
from .algorithms import gradient_descent, softmax
from matrix import Matrix
from random import random


def rand_param():
    return random() - 0.5


class FullConnection(NNFunction):
    def __init__(self, input_channel: int, output_channel: int, learning_rate: float):
        """
        Args:
            input_channel: number of hidden neural node
            output_channel: number of output dimention
        """
        # properties
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.learning_rate = learning_rate
        # cache
        self.x = None
        # internal parameters
        self.W = Matrix.by_generator(output_channel, input_channel, rand_param)
        self.b = Matrix.by_generator(output_channel, 1, rand_param)

    def __repr__(self) -> str:
        return "Ti: " + str(self.input_channel)

    def forward(self, x: Matrix) -> Matrix:
        """ forward inference, implement y = w*x + b

        Args:
            x: input vector, of {input_channel*1} dimention matrix
        Return:
            y: output vector, of {output_channel*1} dimention matrix
        """
        assert x.shape[1] == 1
        y = Matrix.dot(self.W, x) + self.b
        # y = Matrix.dot(self.W, x)
        assert y.shape[1] == 1

        # cache states at current stage
        self.x = x
        return y

    def backward(self, y: Matrix):
        """ backward inference

        Args:
            y: input partial derivative, of {output_channel*1}
               dimention matrix
        Return:
            x: output partial derivative, of {output_channel*1}
               dimention matrix
        """
        assert y.shape == (self.output_channel, 1)
        J_W = self.x.T
        assert J_W.shape == (1, self.input_channel)
        diff_W = Matrix.dot(y, J_W)
        diff_b = y
        # err
        err = Matrix.dot(self.W.T, y)
        # update parameters
        self.W = gradient_descent(self.W, diff_W, self.learning_rate)
        self.b = gradient_descent(self.b, diff_b, self.learning_rate)
        # transfer derivative
        return err
