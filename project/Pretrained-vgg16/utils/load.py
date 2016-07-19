# -*- coding: utf-8 -*-
# Data loading, caching, normalizing utils
import numpy as np
import os
import glob
import cv2
import math
import pickle
import datetime
import pandas as pd

from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from keras.models import model_from_json


np.random.seed(2016)
DATA_DIR = 'D:/hlcv-project/imgs'
DRIVER_FILE = os.path.join(DATA_DIR, 'driver_imgs_list.csv')
CACHE_DIR = 'cache'
SAVED_MODEL_DIR = 'saved_models'
SUBMISSION_DIR = 'subm'

def get_im(path, img_rows, img_cols, color=True):
    if not color:
        # grey
        img = cv2.imread(path, 0)
    elif color:
        # RGB
        img = cv2.imread(path)
    # Reduce size
    resized = cv2.resize(img, (img_cols, img_rows))
    return resized

def get_driver_data():
    dr = dict()
    path = os.path.join(DRIVER_FILE)
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while True:
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
    f.close()
    return dr


def load_train(img_rows, img_cols, color=True):
    X_train = []
    y_train = []
    driver_id = []

    driver_data = get_driver_data()

    print('Read train images')
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join(DATA_DIR, 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_im(fl, img_rows, img_cols, color)
            X_train.append(img)
            y_train.append(j)
            driver_id.append(driver_data[flbase])

    unique_drivers = sorted(list(set(driver_id)))
    print('Unique drivers: {}'.format(len(unique_drivers)))
    print(unique_drivers)
    return X_train, y_train, driver_id, unique_drivers


def load_test(img_rows, img_cols, color=True):
    print('Read test images')
    path = os.path.join(DATA_DIR, 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files) / 10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = get_im(fl, img_rows, img_cols, color)
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total % thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    return X_test, X_test_id


def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory doesnt exists')


def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'rb')
        data = pickle.load(file)
    return data


def save_model(model, model_name):
    print('Saving the model...')
    json_string = model.to_json()
    open(os.path.join(SAVED_MODEL_DIR, '{}_architecture.json'.format(model_name)), 'w').write(json_string)
    model.save_weights(os.path.join(SAVED_MODEL_DIR, '{}_model_weights.h5'.format(model_name)), overwrite=True)


def read_model(model_name):
    print('Loading saved model...')
    model = model_from_json(open(os.path.join(SAVED_MODEL_DIR, '{}_architecture.json'.format(model_name))).read())
    model.load_weights(os.path.join(SAVED_MODEL_DIR, '{}_model_weights.h5'.format(model_name)))
    return model


def split_validation_set(train, target, test_size):
    random_state = 51
    X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def create_submission(model_name, predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join(SUBMISSION_DIR, '{}_submission_{}.csv'.format(model_name, suffix))
    result1.to_csv(sub_file, index=False)
    print('Submission file is created.')


def read_and_normalize_train_data(img_rows, img_cols, color=True, cache=False):
    cache_path = os.path.join(CACHE_DIR, 'train_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color) + '.dat')
    if not os.path.isfile(cache_path) or not cache:
        train_data, train_target, driver_id, unique_drivers = load_train(img_rows, img_cols, color)
        cache_data((train_data, train_target, driver_id, unique_drivers), cache_path)
    else:
        print('Restore train data from cache!')
        (train_data, train_target, driver_id, unique_drivers) = restore_data(cache_path)

    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    if not color:
        # grey
        train_data = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    elif color:
        # RGB
        train_data = train_data.transpose((0, 3, 1, 2))
    train_target = np_utils.to_categorical(train_target, 10)
    train_data = train_data.astype('float32')
    train_data /= 255
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, driver_id, unique_drivers


def read_and_normalize_test_data(img_rows, img_cols, color=True, cache=False):
    cache_path = os.path.join(CACHE_DIR, 'test_r_' + str(img_rows) + '_c_' + str(img_cols) + '_t_' + str(color) + '.dat')
    if not os.path.isfile(cache_path) or not cache:
        test_data, test_id = load_test(img_rows, img_cols, color)
        cache_data((test_data, test_id), cache_path)
    else:
        print('Restore test data from cache!')
        (test_data, test_id) = restore_data(cache_path)

    test_data = np.array(test_data, dtype=np.uint8)
    if not color:
        # grey
        test_data = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    elif color:
        # RGB
        test_data = test_data.transpose((0, 3, 1, 2))
    test_data = test_data.astype('float32')
    test_data /= 255
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    return test_data, test_id


def dict_to_list(d):
    ret = []
    for i in d.items():
        ret.append(i[1])
    return ret


def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()


def merge_several_folds_geom(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a *= np.array(data[i])
    a = np.power(a, 1/nfolds)
    return a.tolist()


def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    data = []
    target = []
    index = []
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            data.append(train_data[i])
            target.append(train_target[i])
            index.append(i)
    data = np.array(data, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    index = np.array(index, dtype=np.uint32)
    return data, target, index
