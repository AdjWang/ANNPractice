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
