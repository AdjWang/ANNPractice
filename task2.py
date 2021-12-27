from os import path
import sys
import random
from math import sin, cos, pi, log
import gzip
from struct import unpack
from random import randint

from dataset import MINST
from matrix import Matrix
from nnlayer import FullConnection, Model, Sigmoid, softmax, argmax, onehot


def random_choice(elements, x):
    n = len(elements)
    assert x <= n
    idx_list = list(range(n))
    random.shuffle(idx_list)
    choose = [elements[idx] for idx in idx_list[:x]]
    left = [elements[idx] for idx in idx_list[x:]]
    return choose, left


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

    @staticmethod
    def split_dataset(train_image, train_label, ratio):
        assert 0.0 <= ratio <= 1.0
        n = len(train_images)
        train_n = int(n * ratio)
        validate_n = n - train_n
        if train_n == 0:
            return (train_image, train_label), ([], [])
        elif validate_n == 0:
            return ([], []), (train_image, train_label)

        choosed_idx, left_idx = random_choice(range(n), train_n)
        train_image_set = [train_image[idx] for idx in choosed_idx]
        train_label_set = [train_label[idx] for idx in choosed_idx]
        validate_image_set = [train_image[idx] for idx in left_idx]
        validate_label_set = [train_label[idx] for idx in left_idx]
        return (train_image_set, train_label_set),\
            (validate_image_set, validate_label_set)


def train_step(model: Model, x, y):
    assert len(x) == len(y)
    training_data = list(zip(x, y))
    loss = 0.0
    for idx, (x, ground_truth) in enumerate(training_data, 1):
        predict = model.forward(Matrix.by_list([x]).T)
        probability = softmax(predict)

        ground_truth = Matrix.by_list([ground_truth]).T
        # loss
        step_loss = (probability * ground_truth)\
            .sum(key=lambda x: -log(x) if x > 0.0 else 0.0)
        loss += step_loss
        # diff of loss
        diff_loss = probability - ground_truth
        model.backward(diff_loss)
        print(f'\rtrain image: {idx}, step_loss: {step_loss}', end='')

    print('')
    return loss/len(training_data)


def validate(model: Model, x, y):
    assert len(x) == len(y)
    validate_data = list(zip(x, y))

    loss = 0.0
    for idx, (x, ground_truth) in enumerate(validate_data, 1):
        predict = model.forward(Matrix.by_list([x]).T)
        probability = softmax(predict)
        ground_truth = Matrix.by_list([ground_truth]).T
        # loss
        step_loss = (probability * ground_truth)\
            .sum(key=lambda x: -log(x) if x > 0.0 else 0.0)
        loss += step_loss
        print(f'\rvalidate image: {idx}, step_loss: {step_loss}', end='')
    print('')
    return loss/len(validate_data)


def train(model: Model, train_x, train_y, validate_x, validate_y, iter_num=100):
    loss_list = []
    for i in range(1, iter_num+1):
        # train
        train_loss = train_step(model, train_x, train_y)
        validate_loss = validate(model, validate_x, validate_y)
        loss = validate_loss
        loss_list.append(loss)
        print(f'iter: {i}, loss: {loss}')
    return loss_list


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
    vec_labels = onehot(output_nodes)

    train_images = [DataLoader.normalize_image(list(image))
                    for image in train_images]
    train_labels = [vec_labels[i] for i in train_labels]
    # assert len(train_images) == len(train_labels)
    # DEBUG: shrink training set to be faster
    train_images, train_labels = train_images[:50], train_labels[:50]

    # split dataset to trian set and validate set
    (train_images, train_labels), (validate_images, validate_labels) = \
        DataLoader.split_dataset(train_images, train_labels, 0.8)

    # comment to only predict -------------------------------------------------
    # initialize model
    model = Model.new_model(input_nodes, output_nodes, [28*28], learning_rate)

    # train
    print('train')
    loss_list = train(model, train_images, train_labels,
                      validate_images, validate_labels, iter_num)

    # save model to file
    model.dump('./model.dump')
    del model

    try:
        from matplotlib import pyplot as plt
        plt.figure(1)
        ax1 = plt.subplot(111)
        ax1.plot(loss_list)
        ax1.set_title('loss')

        plt.show()
    except Exception as e:
        print(f'failed to draw graph due to: {e}')
    # -------------------------------------------------------------------------

    # load model from file
    model = Model.load('./model.dump')

    # predict
    image_index = randint(0, len(test_images)-1)
    predict_num = predict(model,
                          DataLoader.normalize_image(list(test_images[image_index])))
    label_num = test_labels[image_index]
    print(
        f'image.No: {image_index}, predict: {predict_num}, truth: {label_num}')
