from math import sin, cos, pi
from functools import partial

from matrix import Matrix
from nnlayer import FullConnection, Input, Output, ReLU, onehot, Sigmoid


def forward(layers, x):
    for layer in layers:
        x = layer.forward(x)
    return x


def backward(layers, x):
    for layer in reversed(layers):
        x = layer.backward(x)
    return x


def train(layers, datas, iter_num=1000):
    loss_list = []
    for i in range(1, iter_num+1):
        loss = 0.0
        for data, ground_truth in datas:
            predict = forward(layers, Matrix.by_list([data]).T)
            # print(predict)
            ground_truth = Matrix.by_list([ground_truth]).T
            # loss
            step_loss = ((predict - ground_truth) *
                         (predict - ground_truth)).sum()
            loss += step_loss
            # diff of loss
            # diff_loss = ground_truth - predict
            diff_loss = predict - ground_truth
            backward(layers, diff_loss)

        loss_list.append(loss/len(datas))
        print(f'\riter: {i}, loss: {loss/len(datas)}', end='')
    print('')
    return loss_list


def target_func(x, a, b, c, d):
    assert -pi <= x <= pi
    return a*cos(b*x)+c*sin(d*x)


def generate_train_data(target_func, x_start, x_stop, points):
    step = (x_stop - x_start) / points
    datas = []
    for i in range(points):
        x = x_start + i * step
        datas.append(([x], [target_func(x)]))
    return datas


if __name__ == '__main__':
    points = 100
    batch_size = 1
    io_size = batch_size
    hidden_size = 10
    learning_rate = 0.3
    layers = [Input(batch_size),
              FullConnection(batch_size, hidden_size,
                             learning_rate), Sigmoid(hidden_size),
              FullConnection(hidden_size, hidden_size,
                             learning_rate), Sigmoid(hidden_size),
              FullConnection(hidden_size, hidden_size,
                             learning_rate), Sigmoid(hidden_size),
              FullConnection(hidden_size, batch_size,
                             learning_rate), Sigmoid(batch_size),
              Output(batch_size)]
    ground_truth_generator = partial(target_func, a=1, b=1, c=1, d=1)
    datas = generate_train_data(ground_truth_generator, -pi, pi, points=points)
    # print(datas)
    loss = train(layers, datas, 100)
    print(loss)
