import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import initializers


def conv_block(x, filters, name='layer'):

    x = layers.SeparableConv2D(filters, kernel_size=(3, 3), use_bias=False,
                               depthwise_initializer='he_normal', pointwise_initializer='he_normal',
                               padding='same', name='sep_'+name)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    return x


def conv_pw(x, filters, name='layer'):

    x = layers.Conv2D(filters, (1, 1), padding='same',
                      kernel_initializer='he_normal', name='conv_pw_'+name)(x)
    return x


def upblock(x, mobilenet, layer_name, name='4'):

    y = mobilenet.get_layer(name=layer_name).output
    y = conv_pw(y, 32, name='upblock_'+name)

    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.add([y, x])

    x = conv_block(x, filters=32, name='upblock_'+name+'b')

    return x


def segmentation_model(input_shape=(160, 160, 3), numclass=2):

    input_layer = layers.Input(shape=input_shape)
    mobilenet = MobileNet(weights="imagenet", alpha=0.5,
                          input_tensor=input_layer, include_top=False)

    for layer in mobilenet.layers:
        layer.trainable = True

    # Defining custom decoder which is fast and small
    bn = conv_pw(mobilenet.output, filters=32, name='mobile_bottleneck')

    x = upblock(bn, mobilenet, 'conv_pw_11_relu', name='4')
    x = upblock(x, mobilenet, 'conv_pw_5_relu', name='3')
    x = upblock(x, mobilenet, 'conv_pw_3_relu', name='2')
    x = upblock(x, mobilenet, 'conv_pw_1_relu', name='1')

    x = layers.UpSampling2D(size=(2, 2))(x)

    output = layers.Conv2D(filters=numclass, kernel_size=(
        1, 1), padding='same', activation='softmax', name='final_layer')(x)

    model = Model(inputs=input_layer, outputs=output,
                  name='segmentation_model_mobile')

    return model
