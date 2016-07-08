# Fix a bug in printing SVG
import sys
import os
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
from lasagne.regularization import regularize_layer_params_weighted, l2, l1

import theano
from theano import tensor as T

def ZFTurboNet(pixel,input_var=None):
    l_in = InputLayer(shape=(None, 1, pixel, pixel), input_var=input_var)

    l_conv = ConvLayer(l_in, num_filters=64, filter_size=3, pad=1, nonlinearity=rectify)
    l_convb = ConvLayer(l_conv, num_filters=128, filter_size=3, pad=1, nonlinearity=rectify)
    l_pool = MaxPool2DLayer(l_convb, pool_size=2) # feature maps 12x12
    l_dropout1 = DropoutLayer(l_pool, p=0.5)
    l_convc = ConvLayer(l_dropout1, num_filters=256, filter_size=5, pad=1, nonlinearity=rectify)
    l_pool2 = MaxPool2DLayer(l_convc, pool_size=3) # feature maps 12x12
    l_dropout2 = DropoutLayer(l_pool2, p=0.5)
    l_hidden = DenseLayer(l_dropout2, num_units=512, nonlinearity=rectify)    

    l_out = DenseLayer(l_hidden, num_units=10, nonlinearity=softmax)

    return l_out


def getFunctions(pixel, LR = 0.001):
    X = T.tensor4('X')
    Y = T.ivector('y')

    # set up theano functions to generate output by feeding data through network, any test outputs should be deterministic
    output_layer = ZFTurboNet(pixel,X)
    output_train = lasagne.layers.get_output(output_layer)
    output_test = lasagne.layers.get_output(output_layer, deterministic=True)

    # set up the loss that we aim to minimize, when using cat cross entropy our Y should be ints not one-hot
    loss = lasagne.objectives.categorical_crossentropy(output_train, Y)
    penalty = lasagne.regularization.regularize_layer_params(output_layer, l1) * 5e-4
    loss = loss + penalty
    loss = loss.mean()

    # set up loss functions for validation dataset
    valid_loss = lasagne.objectives.categorical_crossentropy(output_test, Y)
    valid_loss = valid_loss.mean()

    valid_acc = T.mean(T.eq(T.argmax(output_test, axis=1), Y), dtype=theano.config.floatX)

    # get parameters from network and set up sgd with nesterov momentum to update parameters
    params = lasagne.layers.get_all_params(output_layer, trainable=True)
    updates = adam(loss, params, learning_rate=LR)

    # set up training and prediction functions
    train_fn = theano.function(inputs=[X,Y], outputs=loss, updates=updates)
    valid_fn = theano.function(inputs=[X,Y], outputs=[valid_loss, valid_acc])

    # set up prediction function
    predict_proba = theano.function(inputs=[X], outputs=output_test)
    
    return train_fn, valid_fn, predict_proba, output_layer