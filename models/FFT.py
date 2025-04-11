from tensorflow.keras.layers import Input, Dense, Conv1D, Conv1DTranspose
from tensorflow.keras.models import Model

import tensorflow as tf

class Transpose(tf.keras.layers.Layer):
    def call(self, x):
        return tf.transpose(x, perm=[0, 2, 1])


def FFT_model(input_shape, bias=True):
    t = input_shape[0]
    input = Input(input_shape, )
    x = Transpose()(input)
    x = Dense(t * 2, activation='tanh', use_bias=bias)(x)
    x = Dense(t * 4, input_dim=t * 2, activation='tanh', use_bias=bias)(x)
    x = Dense(t, input_dim=t * 4, activation='tanh', use_bias=bias)(x)

    x = Transpose()(x)

    x = Conv1D(128, 5, strides=1, padding='same', activation='tanh', use_bias=bias)(x)
    x = Conv1D(128, 5, strides=1, padding='same', activation='tanh', use_bias=bias)(x)

    output = Conv1DTranspose(input_shape[1], 7, strides=1, padding='same', use_bias=bias)(x)

    model = Model(input, output)
    return model