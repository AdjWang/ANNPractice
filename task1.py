from matplotlib import pyplot as plt

from matrix import Matrix
from nnlayer import FullConnection, Input, Output, ReLU, onehot
from nnlayer.sigmoid import Sigmoid


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
        print(f'iter: {i}')
        loss = 0.0
        for data, ground_truth in datas:
            predict = forward(layers, Matrix.by_list([data]).T)
            print(predict)
            ground_truth = Matrix.by_list([ground_truth]).T
            # loss
            loss += ((predict - ground_truth)*(predict - ground_truth)).sum()
            # diff of loss
            # diff_loss = ground_truth - predict
            diff_loss = predict - ground_truth
            backward(layers, diff_loss)

        loss_list.append(loss/len(datas))
        print('loss: ', loss/len(datas))

    plt.plot(loss_list)
    plt.show()


if __name__ == '__main__':
    layers = [Input(1),
              FullConnection(1, 2, 0.35), Sigmoid(2),
              FullConnection(2, 2, 0.35), Sigmoid(2),
              Output(2)]
    # layers = [Input(1), FullConnection(1, 2, 0.001), Output(2)]
    # layers[1].W = Matrix.by_list(
    #     [[0.20319638, 0.13390837], [1.32103492, 0.34115402]])
    # layers[3].W = Matrix.by_list(
    #     [[0.20058315,  1.2256579], [-0.36491087, -0.15542858]])
    ground_truth = onehot(2)
    datas = [
        ([0.1], ground_truth[0]),
        ([0.2], ground_truth[0]),
        ([0.8], ground_truth[1]),
        ([0.9], ground_truth[1]),
        # ([1, 1], ground_truth[0]),
        # ([2, 2], ground_truth[0]),
        # ([8, 8], ground_truth[1]),
        # ([9, 9], ground_truth[1]),
    ]
    train(layers, datas, 400)
