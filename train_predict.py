import os
import glob
import random
import cv2
import numpy as np
from patchify import patchify, unpatchify
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.layers import BatchNormalization, Input, Activation, Add, GlobalAveragePooling2D, Reshape, Dense, multiply, Permute, maximum, Concatenate, Multiply
from keras import backend as K
from keras import metrics
import tensorflow_addons as tfa


project_path = 'C:\\Users\\18101552\\PycharmProjects\\thesis\\thesis_final'

image_names = glob.glob(project_path + "\\dataset\\train_images\\train\\*.png")
image_names.sort()
images = [cv2.imread(img, 0) for img in image_names]
image_dataset = np.array(images)
image_dataset = np.expand_dims(image_dataset, axis=3)

mask_names = glob.glob(project_path + "\\dataset\\train_masks\\train\\*.png")
mask_names.sort()
masks = [cv2.imread(mask, 0) for mask in mask_names]
mask_dataset = np.array(masks)
mask_dataset = np.expand_dims(mask_dataset, axis=3)

image_dataset = image_dataset / 255.
mask_dataset = mask_dataset / 255.

# X_train, X_test, y_train, y_test = train_test_split(image_dataset, mask_dataset, test_size=0.20, random_state=42)
X_train, X_temp, y_train, y_temp = train_test_split(image_dataset, mask_dataset, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

"""# **Preparing the Model**"""


def attention_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        return d

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    def attention_block(F_g, F_l, F_int):
        g = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
        g = BatchNormalization()(g)
        x = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
        x = BatchNormalization()(x)
        psi = Add()([g, x])
        psi = Activation('relu')(psi)

        psi = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)
        psi = Activation('sigmoid')(psi)

        return Multiply()([F_l, psi])

    inputs = Input(shape=(width, height, input_channels))

    conv1 = conv2d(inputs, filters, conv_layers=conv_layers)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = conv2d(pool1, filters * 2, conv_layers=conv_layers)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv2d(pool2, filters * 4, conv_layers=conv_layers)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = conv2d(pool3, filters * 8, conv_layers=conv_layers)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = conv2d(pool4, filters * 16, conv_layers=conv_layers)

    up6 = deconv2d(conv5, filters * 8)
    conv6 = attention_block(up6, conv4, filters * 8)
    up6 = Concatenate()([up6, conv6])
    conv6 = conv2d(up6, filters * 8, conv_layers=conv_layers)

    up7 = deconv2d(conv6, filters * 4)
    conv7 = attention_block(up7, conv3, filters * 4)
    up7 = Concatenate()([up7, conv7])
    conv7 = conv2d(up7, filters * 4, conv_layers=conv_layers)

    up8 = deconv2d(conv7, filters * 2)
    conv8 = attention_block(up8, conv2, filters * 2)
    up8 = Concatenate()([up8, conv8])
    conv8 = conv2d(up8, filters * 2, conv_layers=conv_layers)

    up9 = deconv2d(conv8, filters)
    conv9 = attention_block(up9, conv1, filters)
    up9 = Concatenate()([up9, conv9])
    conv9 = conv2d(up9, filters, conv_layers=conv_layers)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def base_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        return d

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = conv2d(inputs, filters, conv_layers=conv_layers)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = conv2d(pool1, filters * 2, conv_layers=conv_layers)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv2d(pool2, filters * 4, conv_layers=conv_layers)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = conv2d(pool3, filters * 8, conv_layers=conv_layers)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = conv2d(pool4, filters * 16, conv_layers=conv_layers)

    up6 = deconv2d(conv5, filters * 8)
    up6 = Concatenate()([up6, conv4])
    conv6 = conv2d(up6, filters * 8, conv_layers=conv_layers)

    up7 = deconv2d(conv6, filters * 4)
    up7 = Concatenate()([up7, conv3])
    conv7 = conv2d(up7, filters * 4, conv_layers=conv_layers)

    up8 = deconv2d(conv7, filters * 2)
    up8 = Concatenate()([up8, conv2])
    conv8 = conv2d(up8, filters * 2, conv_layers=conv_layers)

    up9 = deconv2d(conv8, filters)
    up9 = Concatenate()([up9, conv1])
    conv9 = conv2d(up9, filters, conv_layers=conv_layers)

    # Changed sigmoid to sigmoid, also changed output from 1 to 4
    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def dense_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        concats = []

        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        concats.append(d)
        M = d

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(M)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

            concats.append(d)
            M = concatenate(concats)

        return M

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = conv2d(inputs, filters, conv_layers=conv_layers)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2d(pool1, filters * 2, conv_layers=conv_layers)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv2d(pool2, filters * 4, conv_layers=conv_layers)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv2d(pool3, filters * 8, conv_layers=conv_layers)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv2d(pool4, filters * 16, conv_layers=conv_layers)

    up6 = deconv2d(conv5, filters * 8)
    merge6 = concatenate([conv4, up6])
    conv6 = conv2d(merge6, filters * 8, conv_layers=conv_layers)

    up7 = deconv2d(conv6, filters * 4)
    merge7 = concatenate([conv3, up7])
    conv7 = conv2d(merge7, filters * 4, conv_layers=conv_layers)

    up8 = deconv2d(conv7, filters * 2)
    merge8 = concatenate([conv2, up8])
    conv8 = conv2d(merge8, filters * 2, conv_layers=conv_layers)

    up9 = deconv2d(conv8, filters)
    merge9 = concatenate([conv1, up9])
    conv9 = conv2d(merge9, filters, conv_layers=conv_layers)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def inception_unet(filters, output_channels, width=None, height=None, input_channels=1):
    def InceptionModule(inputs, filters):
        tower0 = Conv2D(filters, (1, 1), padding='same')(inputs)
        tower0 = BatchNormalization()(tower0)
        tower0 = Activation('relu')(tower0)

        tower1 = Conv2D(filters, (1, 1), padding='same')(inputs)
        tower1 = BatchNormalization()(tower1)
        tower1 = Activation('relu')(tower1)
        tower1 = Conv2D(filters, (3, 3), padding='same')(tower1)
        tower1 = BatchNormalization()(tower1)
        tower1 = Activation('relu')(tower1)

        tower2 = Conv2D(filters, (1, 1), padding='same')(inputs)
        tower2 = BatchNormalization()(tower2)
        tower2 = Activation('relu')(tower2)
        tower2 = Conv2D(filters, (3, 3), padding='same')(tower2)
        tower2 = Conv2D(filters, (3, 3), padding='same')(tower2)
        tower2 = BatchNormalization()(tower2)
        tower2 = Activation('relu')(tower2)

        tower3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        tower3 = Conv2D(filters, (1, 1), padding='same')(tower3)
        tower3 = BatchNormalization()(tower3)
        tower3 = Activation('relu')(tower3)

        inception_module = concatenate([tower0, tower1, tower2, tower3], axis=3)

        return inception_module

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)

        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = InceptionModule(inputs, filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = InceptionModule(pool1, filters * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = InceptionModule(pool2, filters * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = InceptionModule(pool3, filters * 8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = InceptionModule(pool4, filters * 16)

    up6 = deconv2d(conv5, filters * 8)
    up6 = InceptionModule(up6, filters * 8)
    merge6 = concatenate([conv4, up6], axis=3)

    up7 = deconv2d(merge6, filters * 4)
    up7 = InceptionModule(up7, filters * 4)
    merge7 = concatenate([conv3, up7], axis=3)

    up8 = deconv2d(merge7, filters * 2)
    up8 = InceptionModule(up8, filters * 2)
    merge8 = concatenate([conv2, up8], axis=3)

    up9 = deconv2d(merge8, filters)
    up9 = InceptionModule(up9, filters)
    merge9 = concatenate([conv1, up9], axis=3)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(merge9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def r2_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2, rr_layers=2):
    def recurrent_block(layer_input, filters, conv_layers=2, rr_layers=2):
        convs = []
        for i in range(conv_layers - 1):
            a = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')
            convs.append(a)

        d = layer_input
        for i in range(len(convs)):
            a = convs[i]
            d = a(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        for j in range(rr_layers):
            d = Add()([d, layer_input])
            for i in range(len(convs)):
                a = convs[i]
                d = a(d)
                d = BatchNormalization()(d)
                d = Activation('relu')(d)

        return d

    def RRCNN_block(layer_input, filters, conv_layers=2, rr_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d1 = recurrent_block(d, filters, conv_layers=conv_layers, rr_layers=rr_layers)
        return Add()([d, d1])

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = RRCNN_block(inputs, filters, conv_layers=conv_layers, rr_layers=rr_layers)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = RRCNN_block(pool1, filters * 2, conv_layers=conv_layers, rr_layers=rr_layers)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = RRCNN_block(pool2, filters * 4, conv_layers=conv_layers, rr_layers=rr_layers)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = RRCNN_block(pool3, filters * 8, conv_layers=conv_layers, rr_layers=rr_layers)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = RRCNN_block(pool4, filters * 16, conv_layers=conv_layers, rr_layers=rr_layers)

    conv6 = deconv2d(conv5, filters * 8)
    up6 = concatenate([conv6, conv4])
    up6 = RRCNN_block(up6, filters * 8, conv_layers=conv_layers, rr_layers=rr_layers)

    conv7 = Conv2DTranspose(filters * 4, 3, strides=(2, 2), padding='same')(up6)
    up7 = concatenate([conv7, conv3])
    up7 = RRCNN_block(up7, filters * 4, conv_layers=conv_layers, rr_layers=rr_layers)

    conv8 = Conv2DTranspose(filters * 2, 3, strides=(2, 2), padding='same')(up7)
    up8 = concatenate([conv8, conv2])
    up8 = RRCNN_block(up8, filters * 2, conv_layers=conv_layers, rr_layers=rr_layers)

    conv9 = Conv2DTranspose(filters, 3, strides=(2, 2), padding='same')(up8)
    up9 = concatenate([conv9, conv1])
    up9 = RRCNN_block(up9, filters, conv_layers=conv_layers, rr_layers=rr_layers)

    output_layer_noActi = Conv2D(output_channels, (1, 1), padding="same", activation=None)(up9)
    outputs = Activation('sigmoid')(output_layer_noActi)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def residual_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def residual_block(x, filters, conv_layers=2):
        x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        d = x
        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        x = Add()([d, x])

        return x

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = residual_block(inputs, filters, conv_layers=conv_layers)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = residual_block(pool1, filters * 2, conv_layers=conv_layers)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = residual_block(pool2, filters * 4, conv_layers=conv_layers)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = residual_block(pool3, filters * 8, conv_layers=conv_layers)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = residual_block(pool4, filters * 16, conv_layers=conv_layers)

    conv6 = deconv2d(conv5, filters * 8)
    up6 = concatenate([conv6, conv4])
    up6 = residual_block(up6, filters * 8, conv_layers=conv_layers)

    conv7 = deconv2d(up6, filters * 4)
    up7 = concatenate([conv7, conv3])
    up7 = residual_block(up7, filters * 4, conv_layers=conv_layers)

    conv8 = deconv2d(up7, filters * 2)
    up8 = concatenate([conv8, conv2])
    up8 = residual_block(up8, filters * 2, conv_layers=conv_layers)

    conv9 = deconv2d(up8, filters)
    up9 = concatenate([conv9, conv1])
    up9 = residual_block(up9, filters, conv_layers=conv_layers)

    output_layer_noActi = Conv2D(output_channels, (1, 1), padding="same", activation=None)(up9)
    outputs = Activation('sigmoid')(output_layer_noActi)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def se_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        return d

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    def cse_block(inp, ratio=2):
        init = inp
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = multiply([init, se])
        return x

    def sse_block(inp):
        x = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', use_bias=False)(inp)
        x = multiply([inp, x])

        return x

    def scse_block(inp, ratio=2):
        x1 = cse_block(inp, ratio)
        x2 = sse_block(inp)

        x = maximum([x1, x2])

        return x

    inputs = Input(shape=(width, height, input_channels))

    conv1 = conv2d(inputs, filters, conv_layers=conv_layers)
    conv1 = scse_block(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = conv2d(pool1, filters * 2, conv_layers=conv_layers)
    conv2 = scse_block(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv2d(pool2, filters * 4, conv_layers=conv_layers)
    conv3 = scse_block(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = conv2d(pool3, filters * 8, conv_layers=conv_layers)
    conv4 = scse_block(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = conv2d(pool4, filters * 16, conv_layers=conv_layers)
    conv5 = scse_block(conv5)

    up6 = deconv2d(conv5, filters * 8)
    up6 = Concatenate()([up6, conv4])
    conv6 = conv2d(up6, filters * 8, conv_layers=conv_layers)
    conv6 = scse_block(conv6)

    up7 = deconv2d(conv6, filters * 4)
    up7 = Concatenate()([up7, conv3])
    conv7 = conv2d(up7, filters * 4, conv_layers=conv_layers)
    conv7 = scse_block(conv7)

    up8 = deconv2d(conv7, filters * 2)
    up8 = Concatenate()([up8, conv2])
    conv8 = conv2d(up8, filters * 2, conv_layers=conv_layers)
    conv8 = scse_block(conv8)

    up9 = deconv2d(conv8, filters)
    up9 = Concatenate()([up9, conv1])
    conv9 = conv2d(up9, filters, conv_layers=conv_layers)
    conv9 = scse_block(conv9)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def unetpp(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        return d

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv00 = conv2d(inputs, filters, conv_layers=conv_layers)
    pool0 = MaxPooling2D((2, 2))(conv00)

    conv10 = conv2d(pool0, filters * 2, conv_layers=conv_layers)
    pool1 = MaxPooling2D((2, 2))(conv10)

    conv01 = deconv2d(conv10, filters)
    conv01 = concatenate([conv00, conv01])
    conv01 = conv2d(conv01, filters, conv_layers=conv_layers)

    conv20 = conv2d(pool1, filters * 4, conv_layers=conv_layers)
    pool2 = MaxPooling2D((2, 2))(conv20)

    conv11 = deconv2d(conv20, filters)
    conv11 = concatenate([conv10, conv11])
    conv11 = conv2d(conv11, filters, conv_layers=conv_layers)

    conv02 = deconv2d(conv11, filters)
    conv02 = concatenate([conv00, conv01, conv02])
    conv02 = conv2d(conv02, filters, conv_layers=conv_layers)

    conv30 = conv2d(pool2, filters * 8, conv_layers=conv_layers)
    pool3 = MaxPooling2D((2, 2))(conv30)

    conv21 = deconv2d(conv30, filters)
    conv21 = concatenate([conv20, conv21])
    conv21 = conv2d(conv21, filters, conv_layers=conv_layers)

    conv12 = deconv2d(conv21, filters)
    conv12 = concatenate([conv10, conv11, conv12])
    conv12 = conv2d(conv12, filters, conv_layers=conv_layers)

    conv03 = deconv2d(conv12, filters)
    conv03 = concatenate([conv00, conv01, conv02, conv03])
    conv03 = conv2d(conv03, filters, conv_layers=conv_layers)

    conv40 = conv2d(pool3, filters * 16)

    conv31 = deconv2d(conv40, filters * 8)
    conv31 = concatenate([conv31, conv30])
    conv31 = conv2d(conv31, 8 * filters, conv_layers=conv_layers)

    conv22 = deconv2d(conv31, filters * 4)
    conv22 = concatenate([conv22, conv20, conv21])
    conv22 = conv2d(conv22, 4 * filters, conv_layers=conv_layers)

    conv13 = deconv2d(conv22, filters * 2)
    conv13 = concatenate([conv13, conv10, conv11, conv12])
    conv13 = conv2d(conv13, 2 * filters, conv_layers=conv_layers)

    conv04 = deconv2d(conv13, filters)
    conv04 = concatenate([conv04, conv00, conv01, conv02, conv03], axis=3)
    conv04 = conv2d(conv04, filters, conv_layers=conv_layers)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='sigmoid')(conv04)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def dice_coef(y_true, y_pred, smooth=100):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def create_model(model_name, filters, conv_layers, rr_layers):
    if model_name == 'inception_unet':
        return inception_unet(filters=filters, output_channels=1, width=256, height=256, input_channels=1)
    elif model_name == 'r2_unet':
        if rr_layers is None:
            return r2_unet(filters=filters, output_channels=1, width=256, height=256, input_channels=1, conv_layers=conv_layers)
        else:
            return r2_unet(filters=filters, output_channels=1, width=256, height=256, input_channels=1, conv_layers=conv_layers, rr_layers=rr_layers)
    else:
        if conv_layers is None:
            return globals()[model_name](filters=filters, output_channels=1, width=256, height=256, input_channels=1)
        else:
            return globals()[model_name](filters=filters, output_channels=1, width=256, height=256, input_channels=1, conv_layers=conv_layers)


def train_predict(model_name, filters=16, conv_layers=2, rr_layers=None, learning_rate=0.0001, batch_size=32, epochs=100, threshold=0.75):
    results_sub_folder = f'filters_{filters}_learning_rate_{learning_rate}_batch_size_{batch_size}_epochs_{epochs}_threshold_{threshold}'
    results_sub_folder_path = project_path + '\\results\\' + results_sub_folder

    results_model_name = model_name
    if conv_layers is not None:
        results_model_name = f'{results_model_name}_c{conv_layers}'
    if rr_layers is not None:
        results_model_name = f'{results_model_name}_rr{rr_layers}'
    results_path = results_sub_folder_path + '\\' + results_model_name

    if not os.path.isdir(results_sub_folder_path + '\\'):
        os.mkdir(results_sub_folder_path)

    if not os.path.isdir(results_path + '\\'):
        os.mkdir(results_path)

    if not os.path.isdir(results_path + '\\'):
        os.mkdir(results_path)

    if os.path.isfile(results_path + '\\trained_model.hdf5'):
        print("This model is already trained.")
        return

    model = create_model(model_name, filters, conv_layers, rr_layers)

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=dice_coef_loss,
        metrics=['accuracy', dice_coef, tfa.metrics.F1Score(num_classes=2, average="micro", threshold=threshold), metrics.BinaryIoU(threshold=threshold)]
    )

    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test), shuffle=False)

    model.save(results_path + '\\trained_model.hdf5')

    print("\n\nModel:", model_name)
    print("Parameters:", model.count_params())
    print(f"\nAccuracy: {float(np.round((history.history['val_accuracy'][-1])*100, 2))}%")
    print(f"Dice Coefficient: {float(np.round((history.history['val_dice_coef'][-1])*100, 2))}%")
    print(f"F1-score: {float(np.round((history.history['val_f1_score'][-1])*100, 2))}%")
    print(f"IoU: {float(np.round((history.history['val_binary_io_u'][-1]) * 100, 2))}%\n\n")

    open(results_path + "\\train_log.txt", 'w').close()

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epo = range(1, len(loss) + 1)
    plt.plot(epo, loss, 'y', label='Training loss')
    plt.plot(epo, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation: loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(results_path + '\\training_validation_loss.png')
    plt.show()

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epo, acc, 'y', label='Training accuracy')
    plt.plot(epo, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation: accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(results_path + '\\training_validation_accuracy.png')
    plt.show()

    acc = history.history['dice_coef']
    val_acc = history.history['val_dice_coef']
    plt.plot(epo, acc, 'y', label='Training Dice Coef')
    plt.plot(epo, val_acc, 'r', label='Validation Dice Coef')
    plt.title('Training and validation: Dice Coef')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()
    plt.savefig(results_path + '\\training_validation_dice_coef.png')
    plt.show()

    acc = history.history['f1_score']
    val_acc = history.history['val_f1_score']
    plt.plot(epo, acc, 'y', label='Training F1-score')
    plt.plot(epo, val_acc, 'r', label='Validation F1-score')
    plt.title('Training and validation: F1-score')
    plt.xlabel('Epochs')
    plt.ylabel('F1-score')
    plt.legend()
    plt.savefig(results_path + '\\training_validation_f1_score.png')
    plt.show()

    acc = history.history['binary_io_u']
    val_acc = history.history['val_binary_io_u']
    plt.plot(epo, acc, 'y', label='Training IOU')
    plt.plot(epo, val_acc, 'r', label='Validation IOU')
    plt.title('Training and validation: IOU')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.savefig(results_path + '\\training_validation_iou.png')
    plt.show()

    """# **Predicting on a Test Patch**"""

    test_img_number = random.randint(0, len(X_test)-1)
    test_img = X_test[test_img_number]
    ground_truth = y_test[test_img_number]
    test_img_input = np.expand_dims(test_img, 0)

    prediction = (model.predict(test_img_input)[0, :, :, 0] > threshold).astype(np.uint8)

    plt.figure(figsize=(16, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(232)
    plt.title('Ground Truth')
    plt.imshow(ground_truth[:, :, 0])
    plt.subplot(233)
    plt.title('Prediction')
    plt.imshow(prediction)
    plt.show()

    """# **Predicting on a Test Image**"""
    model = load_model(results_path + '\\trained_model.hdf5', compile=False)

    large_image = cv2.imread(project_path + '\\large_image\\large.png', 0)
    patches = patchify(large_image, (256, 256), step=256)

    predicted_patches = []

    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            single_patch = patches[i, j, :, :]
            single_patch = single_patch / 255.
            single_patch_input = np.expand_dims(single_patch, 0)

            single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > threshold).astype(np.uint8)
            predicted_patches.append(single_patch_prediction)

    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256, 256))

    plt.figure(figsize=(12, 12))
    square = 4
    ix = 1
    for i in range(square):
        for j in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(predicted_patches_reshaped[i, j, :, :])
            ix += 1
    plt.savefig(results_path + '\\prediction_patches.png')
    plt.show()

    reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
    plt.imsave(
        results_path + '\\prediction_reconstructed.png',
        reconstructed_image
    )

    plt.figure(figsize=(28, 14))
    plt.subplot(221)
    plt.title('Image')
    plt.imshow(large_image, cmap='gray')
    plt.subplot(222)
    plt.title('Prediction')
    plt.imshow(reconstructed_image)
    plt.savefig(results_path + '\\result_(image_and_prediction).png')
    plt.show()


train_predict(model_name='residual_unet', filters=16, conv_layers=3, learning_rate=0.0001, batch_size=32, epochs=100, threshold=0.75)
