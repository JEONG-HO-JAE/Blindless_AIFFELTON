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
    pool_outputs = []
    for size in pool_sizes:
        pooled = GlobalAveragePooling2D()(x)
        pooled = Reshape((1, 1, pooled.shape[-1]))(pooled)
        pooled = Conv2D(128, (1, 1), activation='relu', padding='same')(pooled)
        pooled = UpSampling2D(size=(x.shape[1], x.shape[2]))(pooled)
        pool_outputs.append(pooled)

    pyramid_output = concatenate(pool_outputs, axis=-1)
    return pyramid_output

def unet_model(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    # Encoder
    conv1 = inception_module(inputs, [64, 128, 128, 256, 64])
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = inception_module(pool1, [128, 256, 256, 512, 128])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = inception_module(pool2, [256, 512, 512, 1024, 256])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = inception_module(pool3, [512, 1024, 1024, 2048, 512])
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Pyramid Pooling Module
    pyramid_output = pyramid_pooling_module(pool4, [1, 2, 3, 6])

    # Decoder
    up6 = UpSampling2D(size=(2, 2))(pyramid_output)
    
    up7_p = concatenate([pool3, pyramid_pooling_module(up6, [1, 2, 3, 6])], axis=-1)

    up7 = UpSampling2D(size=(2, 2))(up7_p)

    up8_p = concatenate([pool2, pyramid_pooling_module(up7, [1, 2, 3, 6])], axis=-1)

    up8 = UpSampling2D(size=(2, 2))(up8_p)

    up9_p = concatenate([pool1, pyramid_pooling_module(up8, [1, 2, 3, 6])], axis=-1)

    up9 = UpSampling2D(size=(2, 2))(up9_p)

    # Output layer
    output = Conv2D(1, (1, 1), activation='sigmoid')(up9)

    model = Model(inputs=inputs, outputs=output)
    return model

# 모델 생성
model = unet_model()
model.summary()