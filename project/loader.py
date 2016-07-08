import os
import cv2
import glob
import math
import time
import random
import numpy as np
import pandas as pd

from scipy import ndimage
from random import random

from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split

'''
Loading data functions
'''
def load_train_cv(encoder, pixels, num_features):
    #global mean_img
    mean_img = np.zeros((pixels, pixels), dtype='uint16')
    X_train = []
    y_train = []
    print('Read train images')        
    for j in range(10):
        print('Load folder c{}'.format(j))
        path = os.path.join('imgs', 'train', 'c' + str(j), '*.jpg')
        files = glob.glob(path)        
        for fl in files:        	
            img = cv2.imread(fl,0)            
            img = cv2.resize(img, (pixels, pixels))   
            mean_img += img          
            X_train.append(img)
            y_train.append(j)

    # mean img subtraction    
    mean_img = mean_img / len(X_train)    
    for i, img in enumerate(X_train):
        img -= mean_img
        img = np.reshape(img, (1, num_features))      
        X_train[i] = img
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    y_train = encoder.fit_transform(y_train).astype('int32')

    X_train, y_train = shuffle(X_train, y_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1)

    X_train = X_train.reshape(X_train.shape[0], 1, pixels, pixels).astype('float32') / 255.
    X_test = X_test.reshape(X_test.shape[0], 1, pixels, pixels).astype('float32') / 255.

    return X_train, y_train, X_test, y_test, encoder, mean_img

def load_test(pixels, num_features, mean_img):
    print('Read test images')
    path = os.path.join('imgs', 'test', '*.jpg')
    files = glob.glob(path)
    X_test = []
    X_test_id = []
    total = 0
    thr = math.floor(len(files)/10)
    for fl in files:
        flbase = os.path.basename(fl)
        img = cv2.imread(fl,0)
        img = cv2.resize(img, (pixels, pixels))
        
        # mean img subtraction
        #global mean_img
        img = img - mean_img
        
        #img = img.transpose(2, 0, 1)
        img = np.reshape(img, (1, num_features))
        X_test.append(img)
        X_test_id.append(flbase)
        total += 1
        if total%thr == 0:
            print('Read {} images from {}'.format(total, len(files)))

    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)

    X_test = X_test.reshape(X_test.shape[0], 1, pixels, pixels).astype('float32') / 255.

    return X_test, X_test_id