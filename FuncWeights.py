import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Input, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
from tensorflow.keras import backend as K

K.set_image_data_format('channels_last')

import os
import numpy as np
from numpy import genfromtxt
import h5py

WEIGHTS = [
    'conv1', 'bn1', 'conv2', 'bn2', 'conv3', 'bn3',
    'inception_3a_1x1_conv', 'inception_3a_1x1_bn', 'inception_3a_pool_conv', 'inception_3a_pool_bn',
    'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
    'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
    'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
    'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
    'inception_3b_pool_conv', 'inception_3b_pool_bn',
    'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
    'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
    'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
    'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
    'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
    'inception_4a_pool_conv', 'inception_4a_pool_bn',
    'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
    'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
    'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
    'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
    'inception_5a_pool_conv', 'inception_5a_pool_bn',
    'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
    'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
    'inception_5b_pool_conv', 'inception_5b_pool_bn',
    'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
    'dense_layer'
]

conv_shape = {
    'conv1': [64, 3, 7, 7], 'conv2': [64, 64, 1, 1], 'conv3': [192, 64, 3, 3],
    'inception_3a_1x1_conv': [64, 192, 1, 1], 'inception_3a_pool_conv': [32, 192, 1, 1],
    'inception_3a_5x5_conv1': [16, 192, 1, 1], 'inception_3a_5x5_conv2': [32, 16, 5, 5],
    'inception_3a_3x3_conv1': [96, 192, 1, 1], 'inception_3a_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_3x3_conv1': [96, 256, 1, 1], 'inception_3b_3x3_conv2': [128, 96, 3, 3],
    'inception_3b_5x5_conv1': [32, 256, 1, 1], 'inception_3b_5x5_conv2': [64, 32, 5, 5],
    'inception_3b_pool_conv': [64, 256, 1, 1], 'inception_3b_1x1_conv': [64, 256, 1, 1],
    'inception_3c_3x3_conv1': [128, 320, 1, 1], 'inception_3c_3x3_conv2': [256, 128, 3, 3],
    'inception_3c_5x5_conv1': [32, 320, 1, 1], 'inception_3c_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_3x3_conv1': [96, 640, 1, 1], 'inception_4a_3x3_conv2': [192, 96, 3, 3],
    'inception_4a_5x5_conv1': [32, 640, 1, 1, ], 'inception_4a_5x5_conv2': [64, 32, 5, 5],
    'inception_4a_pool_conv': [128, 640, 1, 1], 'inception_4a_1x1_conv': [256, 640, 1, 1],
    'inception_4e_3x3_conv1': [160, 640, 1, 1], 'inception_4e_3x3_conv2': [256, 160, 3, 3],
    'inception_4e_5x5_conv1': [64, 640, 1, 1], 'inception_4e_5x5_conv2': [128, 64, 5, 5],
    'inception_5a_3x3_conv1': [96, 1024, 1, 1], 'inception_5a_3x3_conv2': [384, 96, 3, 3],
    'inception_5a_pool_conv': [96, 1024, 1, 1], 'inception_5a_1x1_conv': [256, 1024, 1, 1],
    'inception_5b_3x3_conv1': [96, 736, 1, 1], 'inception_5b_3x3_conv2': [384, 96, 3, 3],
    'inception_5b_pool_conv': [96, 736, 1, 1], 'inception_5b_1x1_conv': [256, 736, 1, 1],
}

def Conv2DBn(x
             , layer=None
             , cv1_out=None
             , cv1_filter=(1, 1)
             , cv1_strides=(1, 1)
             , cv2_out=None
             , cv2_filter=(3, 3)
             , cv2_strides=(1, 1)
             , padding=None
             ):
    num = '' if cv2_out == None else '1'
    # ჩენელი პირველ ადგილას გადმოტანა არის ასე ვთქვათ მიღებული პრაქტიკა, tf ეგრე მუშაობს
    T = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_last', name=layer + '_conv' + num)(x)
    T = BatchNormalization(axis=-1, epsilon=0.00001, name=layer + '_bn' + num)(T)
    T = Activation('relu')(T)
    if padding == None:
        return T
    T = ZeroPadding2D(padding=padding, data_format='channels_last')(T)
    if cv2_out == None:
        return T
    T = Conv2D(cv2_out, cv2_filter, data_format='channels_last', strides=cv2_strides, name=layer + '_conv' + '2')(T)
    T = BatchNormalization(axis=-1, epsilon=0.00001, name=layer + '_bn' + '2')(T)
    T = Activation('relu')(T)

    return T

def LoadWeights():
    filenames = filter(lambda x: not x.startswith('.'), os.listdir('weights'))
    paths = {}
    weights_dict = {}

    for fnames in filenames:
        paths[fnames.replace('.csv', '')] = 'weights/' + fnames

    for name in WEIGHTS:

        if 'conv' in name:
            # genfromtxt ფუნქცია წამოიღებს ფაილიდან მონაცემებს. ფაილის გზას ვუთითებთ პირველივე არგუმენტად.
            conv_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            # წამოიღებს კოეფიციენტებს, რადგან csv ფაილია საჭიროა მძიმის გამყოფად გადაცემა
            conv_w = np.reshape(conv_w, conv_shape[name])
            # წამოღბულ კოეფიციენტებს უნდა ქონდეთ ისეთი ფორმა როგორიც ნეირონულ ქსელშია
            conv_w = np.transpose(conv_w, (2, 3, 1, 0))
            # ჩენელი, ანუ 3 ფერი უნდა იყოს პირველ ადგილას, ასე ვართ შეთანხმებული, tf ასე მუშაობს.
            conv_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            # ეს სვეტ-მატრიცაა, b  და ამიტო არაფერს საჭიროებს
            weights_dict[name] = [conv_w, conv_b]

        elif 'bn' in name:

            bn_w = genfromtxt(paths[name + '_w'], delimiter=',', dtype=None)
            bn_b = genfromtxt(paths[name + '_b'], delimiter=',', dtype=None)
            bn_m = genfromtxt(paths[name + '_m'], delimiter=',', dtype=None)
            bn_v = genfromtxt(paths[name + '_v'], delimiter=',', dtype=None)
            weights_dict[name] = [bn_w, bn_b, bn_m, bn_v]

        elif 'dense' in name:

            dense_w = genfromtxt('weights/dense_w.csv', delimiter=',', dtype=None)
            dense_w = np.reshape(dense_w, (128, 736))
            dense_w = np.transpose(dense_w, (1, 0))
            dense_b = genfromtxt('weights/dense_b.csv', delimiter=',', dtype=None)
            weights_dict[name] = [dense_w, dense_b]

    return weights_dict

def LoadWeightsFaceNet(Model):
    weights = WEIGHTS
    weights_dict = LoadWeights()

    for name in weights:
        if Model.get_layer(name) != None:
            Model.get_layer(name).set_weights(weights_dict[name])


def ImgToEncoding(image_path, model):
    # სურათის წამოღება სხვანაირადაც შეიძლება მაგრამ ბარემ tf...
    img = tf.keras.preprocessing.image.load_img(image_path)
    # ნორმალიზაციას ვუკეთებთ
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    # ერთით მეტი განზომილება გვჭირდება
    x_train = np.expand_dims(img, axis=0)
    print(x_train.shape)
    # თითოეულ ბათჩზე პროგნოზირებისათვის
    embedding = model.predict_on_batch(x_train)
    return embedding

def RImgToEncoding(image, model):
    # სურათის წამოღება სხვანაირადაც შეიძლება მაგრამ ბარემ tf...
    img = image
    # ნორმალიზაციას ვუკეთებთ
    img = np.around(np.transpose(img, (0, 1, 2)) / 255.0, decimals=12)
    # ერთით მეტი განზომილება გვჭირდება
    x_train = np.expand_dims(img, axis=0)
    print(x_train.shape)
    # თითოეულ ბათჩზე პროგნოზირებისათვის
    embedding = model.predict_on_batch(x_train)
    return embedding
