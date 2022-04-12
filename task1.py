from math import sin, cos, pi
from functools import partial

from matrix import Matrix
from nnlayer import FullConnection, Model, Sigmoid, LinearMapper


def train(model: Model, x, y, iter_num=100):
    assert len(x) == len(y)
    training_data = list(zip(x, y))
    loss_list = []
    predict_list = []
    for i in range(1, iter_num+1):
        loss = 0.0
        for x, ground_truth in training_data:
            predict = model.forward(Matrix.by_list([x]).T)
            predict_list.append(predict)
            # print(predict)
            ground_truth = Matrix.by_list([ground_truth]).T
            # loss
            step_loss = ((predict - ground_truth) *
                         (predict - ground_truth)).sum()
            loss += step_loss
            # diff of loss
            diff_loss = predict - ground_truth
            model.backward(diff_loss)

        loss_list.append(loss/len(training_data))
        print(f'\riter: {i}, loss: {loss/len(training_data)}', end='')
    print('')
    return loss_list, predict_list


def predict(model, x):
    return model.forward(Matrix.by_list([x]).T).T[0]


def target_func(x, a, b, c, d):
    return a*cos(b*x)+c*sin(d*x)


def generate_x(start, stop, points):
    step = (stop - start) / points
    return [start + i * step for i in range(points)]


def generate_y(target_func, x_list):
    return [target_func(x) for x in x_list]


if __name__ == '__main__':
    # config
    points = 50
    x_range = (-pi, pi)  # [-pi, pi)
    a, b, c, d = 1, 1, 1, 1  # a*cos(b*x)+c*sin(d*x)
    iter_num = 100
    learning_rate = 0.6

    # initialize model
    node_num = points
    model = Model([FullConnection(node_num, node_num, learning_rate),
                   Sigmoid(node_num),
                   FullConnection(node_num, node_num, learning_rate),
                   Sigmoid(node_num),
                   ])
    # generate training data
    x = generate_x(x_range[0], x_range[1], points)
    y = generate_y(partial(target_func, a=a, b=b, c=c, d=d), x)
    data_mapper = LinearMapper()
    mapped_y = data_mapper.mapping(y)
    # train
    loss, _ = train(model, [x], [mapped_y], iter_num)
    # predict
    predict_y = predict(model, x)
    revmapped_y = data_mapper.revmapping(predict_y)

    try:
        from matplotlib import pyplot as plt
        plt.figure(1)
        ax1 = plt.subplot(121)
        ax1.plot(loss)
        ax1.set_title('loss')

        ax2 = plt.subplot(122)
        ax2.plot(y, '-', revmapped_y, '+')
        ax2.set_title('curve')
        plt.show()
    except Exception as e:
        print(f'failed to draw graph due to: {e}')
