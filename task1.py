from matplotlib import pyplot as plt

from matrix import Matrix
from nnlayer import FullConnection, Input, Output, ReLU, onehot

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
        loss_mat = None
        for data, ground_truth in datas:
            result = forward(layers, Matrix.by_list([data]).T)
            ground_truth = Matrix.by_list([ground_truth]).T
            loss_mat = result - ground_truth
            backward(layers, Matrix.by_diag(loss_mat.T[0]))
        
        loss = loss_mat.sum(key=abs) / (loss_mat.shape[0]*loss_mat.shape[1])
        loss_list.append(loss)
        print('loss: ', loss)

    plt.plot(loss_list)
    plt.show()

if __name__ == '__main__':
    layers = [Input(1), FullConnection(1, 2, 0.001), ReLU(2), Output(2)]
    ground_truth = onehot(2)
    datas = [
        ([0], ground_truth[0]),
        ([1], ground_truth[0]),
        ([8], ground_truth[1]),
        ([9], ground_truth[1]),
    ]
    train(layers, datas, 40)
    print(layers[1].W)