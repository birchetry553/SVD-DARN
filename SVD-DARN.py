import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    GlobalAveragePooling2D,
    Dense,
    Add,
    Multiply,
    Activation,
    BatchNormalization,
    GlobalMaxPooling2D,
    Reshape,
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
import numpy as np


def spectral_se_block(input_tensor, ratio=2): 
    channels = K.int_shape(input_tensor)[-1]
    se_shape = (1, 1, channels)
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(
        channels // ratio, activation="relu", use_bias=False
    )(se)
    se = Dense(channels, activation="sigmoid", use_bias=False)(se)
    return Multiply()([input_tensor, se])


def spatial_se_block(input_tensor):
    se = Conv2D(
        1, (1, 1), activation="sigmoid", use_bias=False, kernel_regularizer=regularizers.l2(0.0001), kernel_initializer="he_normal"
    )(input_tensor)
    return Multiply()([input_tensor, se])


def ssse_block(input_tensor, ratio=2):
    spectral_se = spectral_se_block(input_tensor, ratio)
    spatial_se = spatial_se_block(input_tensor)
    return Add()([spectral_se, spatial_se])


def res_block_ssse(input_tensor, filters, kernel_size, ratio=2):
    
    f1, f2, f3 = filters
    x = Conv2D(f1, (1, 1), kernel_regularizer=regularizers.l2(0.0001), kernel_initializer="he_normal")(input_tensor)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f2, kernel_size, padding="same", kernel_regularizer=regularizers.l2(0.0001), kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(f3, (1, 1), kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)

    x = ssse_block(x, ratio)

    shortcut = Conv2D(f3, (1, 1), kernel_regularizer=regularizers.l2(0.0001), kernel_initializer="he_normal")(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    # Add shortcut to the main path
    x = Add()([x, shortcut])
    x = Activation("relu")(x)

    return x

def SSSERN(input_shape, num_classes, num_res_blocks=2):
   
    input_layer = Input(shape=input_shape)
    x = Conv2D(
        128,
        (1, 1),
        padding="same",
        kernel_regularizer=regularizers.l2(0.0001),
        kernel_initializer="he_normal",
    )(input_layer)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Stack of SSSE Residual Blocks
    for _ in range(num_res_blocks):
        x = res_block_ssse(x, filters=(32, 32, 128), kernel_size=(3, 3))

    # Global Pooling and Classification Head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)  # Add dropout here (0.5 = 50% drop rate)
    output_layer = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input_layer, outputs=output_layer, name="SSSERN")
    return model