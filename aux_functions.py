import tensorflow as tf
import numpy as np


def load_mnist_dataset():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train_label), (x_test, y_test_label) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.reshape(-1, 28, 28, 1), x_test.reshape(-1, 28, 28, 1)
    y_train = np.zeros((len(y_train_label), 10))
    y_train[range(len(y_train_label)), y_train_label] = 1
    y_test = np.zeros((len(y_test_label), 10))
    y_test[range(len(y_test_label)), y_test_label] = 1
    return x_train, y_train, x_test, y_test


def load_cifar10_dataset():
    cifar10 = tf.keras.datasets.cifar10
    (x_train, y_train_label), (x_test, y_test_label) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train_label = y_train_label.reshape(-1)
    y_test_label = y_test_label.reshape(-1)
    y_train = np.zeros((len(y_train_label), 10))
    y_train[range(len(y_train_label)), y_train_label] = 1
    y_test = np.zeros((len(y_test_label), 10))
    y_test[range(len(y_test_label)), y_test_label] = 1
    return x_train, y_train, x_test, y_test

