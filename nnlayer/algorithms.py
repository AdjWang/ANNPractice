from typing import List
from math import pow, e, log

from matrix import Matrix


def gradient_descent(W: Matrix, diff_W: Matrix, learning_rate: float) -> Matrix:
    delta = Matrix.mul(diff_W, learning_rate)
    return W - delta


def softmax(x: Matrix) -> Matrix:
    assert x.shape[1] == 1
    summary = sum([pow(e, yi) for yi in x.T[0]])
    return Matrix.by_list([[pow(e, yi)/summary for yi in x.T[0]]]).T


def onehot(type_num: int) -> List[List[int]]:
    result = []
    for _ in range(type_num):
        result.append([0]*type_num)

    for i in range(type_num):
        result[i][i] = 1
    return result


def cross_entropy_loss(y: Matrix, t: Matrix) -> float:
    return -(y.apply(log) * t).sum()

class LinearMapper:
    """ Map data to fit the range [0, 1] of sigmoid. """
    def __init__(self) -> None:
        self.alpha = 1.0
        self.bias = 0.0

    def mapping(self, x: List[float]) -> List[float]:
        """ map x [a, b] to [0, 1] """
        min_num = min(x)
        max_num = max(x)
        alpha = 1.0 / (max_num - min_num) * 0.9
        bias = -min_num * alpha
        # record
        self.alpha, self.bias = alpha, bias
        return [alpha * i + bias for i in x]

    def revmapping(self, x: List[float]) -> List[float]:
        """ rev map x [0, 1] to [a, b] """
        return [(i - self.bias) / self.alpha for i in x]