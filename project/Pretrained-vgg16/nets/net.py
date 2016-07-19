# -*- coding: utf-8 -*-
import numpy as np
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD

np.random.seed(2016)

NUM_CLASSES = 10
CONV_SIZE = 3
POOL_SIZE = 2


def baseline(img_rows, img_cols, color=True):
    input_shape = 3 if color else 1

    model = Sequential()
    model.add(Convolution2D(64, CONV_SIZE, CONV_SIZE,
                            border_mode='same',
                            input_shape=(input_shape, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, CONV_SIZE, CONV_SIZE,
                            border_mode='same', init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)))
    model.add(Dropout(0.5))
    model.add(Convolution2D(256, CONV_SIZE, CONV_SIZE,
                            border_mode='same', init='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(POOL_SIZE, POOL_SIZE)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


'''
Runs vgg-16 pre-trained model.
Download the model from https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3
Color_type = 3 --> RGB
'''
def vgg_std16_model(img_rows, img_cols, color=True, path='data/vgg16_weights.h5'):
    input_shape = 3 if color else 1
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(input_shape,
                                                 img_rows, img_cols)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    model.load_weights(path)

    # Code above loads pre-trained data and
    model.layers.pop()
    model.add(Dense(10, activation='softmax'))
    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
