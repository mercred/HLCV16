'''
Doesn't run through kaggle scripts because of the 20 min limit. 
Download the code, expand the network, add more iterations
'''

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


# Fix a bug in printing SVG
import sys
if sys.platform == 'win32':
    print("Monkey-patching pydot")
    import pydot

    def force_find_graphviz(graphviz_root):
        binpath = os.path.join(graphviz_root, 'bin')
        programs = 'dot twopi neato circo fdp sfdp'
        def helper():
            for prg in programs.split():
                if os.path.exists(os.path.join(binpath, prg)):
                    yield ((prg, os.path.join(binpath, prg)))
                elif os.path.exists(os.path.join(binpath, prg+'.exe')):
                    yield ((prg, os.path.join(binpath, prg+'.exe')))
        progdict = dict(helper())
        return lambda: progdict

    pydot.find_graphviz = force_find_graphviz('c:/Program Files (x86)/Graphviz2.34/')

import lasagne
from lasagne.layers import helper
from lasagne.updates import adam
from lasagne.nonlinearities import rectify, softmax
from lasagne.layers import InputLayer, MaxPool2DLayer, DenseLayer, DropoutLayer, helper
from lasagne.layers import Conv2DLayer as ConvLayer

import theano
from theano import tensor as T
from loader import load_train_cv, load_test
from net_functions import getFunctions

PIXELS = 24
imageSize = PIXELS * PIXELS
num_features = imageSize 


'''
Lasagne Model ZFTurboNet and Batch Iterator
'''
def iterate_minibatches(inputs, targets, batchsize):
    assert len(inputs) == len(targets)
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]

"""
Set up all theano functions
"""
BATCHSIZE = 32
MAX_WORSENING_EPOCHE = 3;
train_fn, valid_fn, predict_proba, output_layer = getFunctions(PIXELS);

'''
load training data and start training
'''
encoder = LabelEncoder()

# load the training and validation data sets
train_X, train_y, valid_X, valid_y, encoder, mean_img = load_train_cv(encoder, PIXELS, num_features)
print('Train shape:', train_X.shape, 'Test shape:', valid_X.shape)

# load data
X_test, X_test_id = load_test(PIXELS, num_features, mean_img)
best_val_acc = 0;
worsening_epoche = 0;
epoch = 1


# loop over training functions for however many iterations, print information while training
try:
    #for epoch in range(ITERS):
    while True:
        # do the training
        start = time.time()
        # training batches
        train_loss = []
        for batch in iterate_minibatches(train_X, train_y, BATCHSIZE):
            inputs, targets = batch
            train_loss.append(train_fn(inputs, targets))
        train_loss = np.mean(train_loss)
        # validation batches
        valid_loss = []
        valid_acc = []
        for batch in iterate_minibatches(valid_X, valid_y, BATCHSIZE):
            inputs, targets = batch
            valid_eval = valid_fn(inputs, targets)
            valid_loss.append(valid_eval[0])
            valid_acc.append(valid_eval[1])
        valid_loss = np.mean(valid_loss)
        valid_acc = np.mean(valid_acc)
        # get ratio of TL to VL
        ratio = train_loss / valid_loss
        end = time.time() - start
        # print training details
        print('iter:', epoch, '| TL:', np.round(train_loss,decimals=3), '| VL:', np.round(valid_loss, decimals=3), '| Vacc:', np.round(valid_acc, decimals=3), '| Ratio:', np.round(ratio, decimals=2), '| Time:', np.round(end, decimals=1))
        
        epoch += 1
        #save network parameters if reach better acc
        if best_val_acc < valid_acc:
            with open('net.net', "wb") as f:
                np.save(f, lasagne.layers.get_all_param_values(output_layer))
            best_val_acc = valid_acc
            worsening_epoche = 0
        else:   
            worsening_epoche += 1
            if worsening_epoche == MAX_WORSENING_EPOCHE:
                break
except KeyboardInterrupt:
    pass
    
valid_loss = []
valid_acc = []
for batch in iterate_minibatches(train_X, train_y, BATCHSIZE):
    inputs, targets = batch
    valid_eval = valid_fn(inputs, targets)
    valid_loss.append(valid_eval[0])
    valid_acc.append(valid_eval[1])
valid_loss = np.mean(valid_loss)
valid_acc = np.mean(valid_acc)
# get ratio of TL to VL
ratio = train_loss / valid_loss
end = time.time() - start
# print training details
print('Finall Tacc:', np.round(valid_acc, decimals=3))


'''
Make Submission
'''

#make predictions
print('Making predictions')
PRED_BATCH = 2
def iterate_pred_minibatches(inputs, batchsize):
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

predictions = []
for pred_batch in iterate_pred_minibatches(X_test, PRED_BATCH):
    predictions.extend(predict_proba(pred_batch))

predictions = np.array(predictions)

print('pred shape')
print(predictions.shape)

print('Creating Submission')
def create_submission(predictions, test_id):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    result1.to_csv('submission_ZFTurboNet.csv', index=False)

create_submission(predictions, X_test_id)
