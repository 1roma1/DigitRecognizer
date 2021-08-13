import numpy as np
import tensorflow as tf

def preproc_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = tf.keras.utils.to_categorical(y_train, num_classes=None)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=None)

    return x_train, y_train, x_test, y_test