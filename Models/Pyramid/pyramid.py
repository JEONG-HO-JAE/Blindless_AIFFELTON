from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *

def inception_module(x, filters):
    conv1x1 = Conv2D(filters[0], (1, 1), activation='relu', padding='same')(x)
    conv3x3 = Conv2D(filters[1], (3, 3), activation='relu', padding='same')(x)
    conv_double_3x3 = Conv2D(filters[2], (1, 1), activation='relu', padding='same')(x)
    conv_double_3x3 = Conv2D(filters[3], (3, 3), activation='relu', padding='same')(conv_double_3x3)

    inception_block = concatenate([conv1x1, conv3x3, conv_double_3x3], axis=-1)
    inception_block = Conv2D(filters[4], (1, 1), activation='relu', padding='same')(inception_block)

    return inception_block

def pyramid_pooling_module(x, pool_sizes):
    i = 0
    pool_outputs = []
    for size in pool_sizes:
        pooled = AveragePooling2D(pool_size=(size, size))(x)
        pooled = Conv2D(32, (1, 1), activation='relu', padding='same')(pooled)
        pooled = UpSampling2D(size=(2**i, 2**i))(pooled)
        pool_outputs.append(pooled)
        i+=1
    pool_outputs.append(x)
    pyramid_output = concatenate(pool_outputs, axis=-1)
    return pyramid_output


def unet_model(input_shape=(512, 512, 1)):
    inputs = Input(input_shape)

    # Encoder
    conv1 = inception_module(inputs, [8, 16, 16, 32, 16])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = inception_module(pool1, [16, 32, 32, 64, 32])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = inception_module(pool2, [32, 64, 64, 128, 64])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = inception_module(pool3, [64, 128, 128, 256, 128])
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Pyramid Pooling Module
    pyramid_output = pyramid_pooling_module(pool4, [1, 2, 4, 8])

    # Decoder
    up6 = UpSampling2D(size=(2, 2))(pyramid_output)

    up7_p = concatenate([pool3, pyramid_pooling_module(up6, [1, 2, 4, 8])], axis=-1)

    up7 = UpSampling2D(size=(2, 2))(up7_p)

    up8_p = concatenate([pool2, pyramid_pooling_module(up7, [1, 2, 4, 8])], axis=-1)

    up8 = UpSampling2D(size=(2, 2))(up8_p)

    up9_p = concatenate([pool1, pyramid_pooling_module(up8, [1, 2, 4, 8])], axis=-1)

    up9 = UpSampling2D(size=(2, 2))(up9_p)

    # Output layer
    output = Conv2D(1, (1, 1), activation='sigmoid')(up9)

    model = Model(inputs=inputs, outputs=output)
    return model