from tensorflow.keras.layers import Input, Dense, Reshape, Conv1D, BatchNormalization, \
    MaxPooling1D, AveragePooling1D, GlobalAveragePooling1D, ReLU, concatenate, Concatenate
from tensorflow.keras.models import Model
import tensorflow as tf

def densenet(input_shape, output_shape, f=32):
    def bn_rl_conv(x, filter_, k=1, s=1):
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv1D(filter_, k, strides=s, padding='same')(x)
        return x

    def dense_block(x, r):
        for _ in range(r):
            y = bn_rl_conv(x, 4 * f)
            y = bn_rl_conv(y, f, 3)
            x = Concatenate(axis=-1)([y,x])
        return x

    def transition_layer(x):
        x = bn_rl_conv(x, x.shape.as_list()[-1] //2)
        x = AveragePooling1D(2, strides=2, padding='same')(x)
        return x

    input = Input(input_shape)
    #input = tf.keras.layers.Normalization()(input)
    x = Conv1D(64, 7, strides=2, padding='same')(input)
    x = MaxPooling1D(3, strides=2, padding='same')(x)

    for r in [6, 12, 24, 16]:
        d = dense_block(x, r)
        x = transition_layer(d)

    x = GlobalAveragePooling1D()(d)
    #x = Flatten()(d)
    #x = tf.math.reduce_mean(d, axis=1)
    x = Dense(output_shape[0]*output_shape[-1], activation='relu')(x)
    output = Reshape(output_shape)(x)

    model = Model(inputs=[input], outputs=[output])  # input shape (1024,2) output shape (1024,1)
    return model