from tensorflow import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *

def transition_layer(x, n_filters):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(n_filters, 3, padding='same', activation='relu', kernel_initializer='he_normal')(x)
    return x


def dense_block(input_tensor, nb_filter):
    x1 = transition_layer(input_tensor, nb_filter)
    add1 = concatenate([x1, input_tensor], axis=-1)
    x2 = transition_layer(add1, nb_filter)
    add2 = concatenate([input_tensor, x2], axis=-1)
    return add2

def build_model(input_shape=(64, 64, 1)):
    inputs = Input(input_shape)
    # Contracting Path
    conv1 = Conv2D(48, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = dense_block(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = dense_block(pool1, 240)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = dense_block(pool2, 64)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = dense_block(pool3, 128)

    # Expanding Path
    up6 = Conv2DTranspose(64, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(conv4)
    merge6 = concatenate([conv3, up6], axis=3)
    # conv6 = dense_block(merge6, 64)
    up7 = Conv2DTranspose(240, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(merge6)
    merge7 = concatenate([conv2, up7], axis=3)
    # conv7 = dense_block(merge7, 240)
    up8 = Conv2DTranspose(64, 2, activation='relu', strides=(2, 2), kernel_initializer='he_normal')(merge7)
    merge8 = concatenate([conv1, up8], axis=3)
    # conv8 = dense_block(merge8, 64)
    conv9 = Conv2D(1, 1, activation='sigmoid')(merge8)

    model = Model(inputs=inputs, outputs=conv9)
    return model