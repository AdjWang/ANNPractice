from os import path
import sys
from math import sin, cos, pi, log
import gzip
from struct import unpack
from random import randint

from dataset import MINST
from matrix import Matrix
from nnlayer import FullConnection, Model, Sigmoid, softmax, argmax, onehot


class DataLoader:
    @staticmethod
    def read_gzip(filepath):
        with gzip.open(filepath, 'rb') as f:
            b_content = f.read()
        return b_content

    @staticmethod
    def load_train_image(filepath):
        b_content = DataLoader.read_gzip(filepath)
        content = unpack('>iiii'+f'{28*28}s'*60000, b_content)
        return content[4:]

    @staticmethod
    def load_test_image(filepath):
        b_content = DataLoader.read_gzip(filepath)
        content = unpack('>iiii'+f'{28*28}s'*10000, b_content)
        return content[4:]

    @staticmethod
    def load_train_label(filepath):
        b_content = DataLoader.read_gzip(filepath)
        content = unpack('>ii'+'b'*60000, b_content)
        return content[2:]

    @staticmethod
    def load_test_label(filepath):
        b_content = DataLoader.read_gzip(filepath)
        content = unpack('>ii'+'b'*10000, b_content)
        return content[2:]

    @staticmethod
    def normalize_image(vec):
        return [v/255 for v in vec]


def train(model: Model, x, y, iter_num=100):
    assert len(x) == len(y)
    training_data = list(zip(x, y))
    loss_list = []
    predict_list = []
    for i in range(1, iter_num+1):
        loss = 0.0
        for idx, (x, ground_truth) in enumerate(training_data, 1):
            predict = model.forward(Matrix.by_list([x]).T)
            probability = softmax(predict)
            predict_list.append(argmax(probability.T[0]))

            ground_truth = Matrix.by_list([ground_truth]).T
            # loss
            step_loss = (probability * ground_truth)\
                .sum(key=lambda x: -log(x) if x > 0.0 else 0.0)
            loss += step_loss
            # diff of loss
            diff_loss = probability - ground_truth
            model.backward(diff_loss)
            print(f'\rimage: {idx}, step_loss: {step_loss}', end='')

        print('')
        loss_list.append(loss/len(training_data))
        print(f'iter: {i}, loss: {loss/len(training_data)}')
    return loss_list, predict_list


def predict(model, x):
    probability = softmax(model.forward(Matrix.by_list([x]).T)).T[0]
    return argmax(probability)


if __name__ == '__main__':
    # config
    dataset_path = path.join(path.dirname(sys.argv[0]), 'dataset')
    input_nodes = 28 * 28
    hidden_nodes = 28 * 28
    output_nodes = 10
    iter_num = 10
    learning_rate = 0.3

    # initialize model
    model = Model.new_model(input_nodes, output_nodes, [28*28], learning_rate)

    # load data
    print('verify dataset')
    MINST.verify(dataset_path)

    print('load data')
    train_images = DataLoader.load_train_image(
        path.join(dataset_path, 'train-images-idx3-ubyte.gz'))
    test_images = DataLoader.load_test_image(
        path.join(dataset_path, 't10k-images-idx3-ubyte.gz'))
    train_labels = DataLoader.load_train_label(
        path.join(dataset_path, 'train-labels-idx1-ubyte.gz'))
    test_labels = DataLoader.load_test_label(
        path.join(dataset_path, 't10k-labels-idx1-ubyte.gz'))

    # generate training data
    vec_labels = onehot(10)

    train_images = [DataLoader.normalize_image(list(image))
                    for image in train_images]
    train_labels = [vec_labels[i] for i in train_labels]
    # assert len(train_images) == len(train_labels)
    # DEBUG: shrink training set to be faster
    train_images, train_labels = train_images[:10], train_labels[:10]

    # train
    print('train')
    loss, _ = train(model, train_images, train_labels, iter_num)

    # predict
    image_index = randint(0, len(test_images)-1)
    predict_num = predict(model,
                          DataLoader.normalize_image(list(test_images[image_index])))
    label_num = test_labels[image_index]
    print(f'image.No: {image_index},\
          predict: {predict_num}, truth: {label_num}')
