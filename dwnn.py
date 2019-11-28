#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Feb 27 2019

@author: yzj
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import keras as K
from keras.layers import Input, Dense, Activation, Flatten, Dropout,concatenate, BatchNormalization
from keras.layers.convolutional import Convolution2D, ZeroPadding2D,MaxPooling2D, UpSampling2D, AveragePooling2D
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.regularizers import l1_l2
import random
import numpy as np
import scipy.io as sio  
import sys
import gzip
import matplotlib.pyplot as plt

def dwmlp_mt_en(Xt=None,Yt=None, weights=None, numofc=None, input_tensor=None, 
                   include_top=True,tin=0):
# deep and wide neural networks with multi-task
    nwidth = 512
    numofdim = Xt.shape[2]
    regpara = 0.0000
    dopara = 0.65
    input_img = Input(shape=(1,numofdim,1))

    x = Flatten()(input_img)
    y = Dense(nwidth, activation='relu', name='hidden00')(x)
    y = BatchNormalization()(y)
    y1 = Dropout(dopara)(y)
    y = Dense(nwidth, activation='relu', name='hidden01')(y1)
    y = BatchNormalization()(y)
    y2 = Dropout(dopara)(y)
    y = Dense(nwidth, activation='relu', name='hidden02')(y2)
    y = BatchNormalization()(y)
    y3 = Dropout(dopara)(y)
    y = Dense(nwidth, activation='relu', name='hidden03')(y3)
    y = BatchNormalization()(y)
    y4 = Dropout(dopara)(y)
    x = concatenate([x,y1,y2,y3,y4],axis=1)
    x = Dense(1024, activation='sigmoid', name='hidden70')(x)
    x = Dropout(dopara)(x)
    
    # Here 3 tasks with 6 outputs
    out1 = Dense(2, activation='softmax', name='output0')(x)
    out2 = Dense(2, activation='softmax', name='output1')(x)
    out3 = Dense(2, activation='softmax', name='output2')(x)
    out = concatenate([out1,out2,out3],axis=1)
    
    model = Model(input_img,out)
    model.summary()
    return model

def dwmlp_st(Xt=None,Yt=None, weights=None, numofc=None, input_tensor=None, 
                   include_top=True):
# dwnn with single task
    nwidth = 512
    numofdim = Xt.shape[2]
    regpara = 0.0000
    dopara = 0.65
    input_img = Input(shape=(1,numofdim,1))

    x = Flatten()(input_img)
    y = Dense(nwidth, activation='relu', name='hidden00')(x)
    y = BatchNormalization()(y)
    y1 = Dropout(dopara)(y)
    y = Dense(nwidth, activation='relu', name='hidden01')(y1)
    y = BatchNormalization()(y)
    y2 = Dropout(dopara)(y)
    y = Dense(nwidth, activation='relu', name='hidden02')(y2)
    y = BatchNormalization()(y)
    y3 = Dropout(dopara)(y)
    y = Dense(nwidth, activation='relu', name='hidden03')(y3)
    y = BatchNormalization()(y)
    y4 = Dropout(dopara)(y)
    x = concatenate([x,y1,y2,y3,y4],axis=1)
    x = Dense(1024, activation='sigmoid', name='hidden6')(x)
    x = Dropout(dopara)(x)
    
    # binary classification
    out = Dense(2, activation='softmax', name='output')(x)
    
    model = Model(input_img,out)
    model.summary()
    return model

def dnn_mt_en(Xt=None,Yt=None, weights=None, numofc=None, input_tensor=None, 
                   include_top=True):
# conventional dnn with multi-task
    nwidth = 512
    numofdim = Xt.shape[2]
    regpara = 0.0000
    dopara = 0.65
    input_img = Input(shape=(1,numofdim,1))

    x = Flatten()(input_img)
    y = Dense(nwidth, activation='relu', name='hidden00')(x)
    y = BatchNormalization()(y)
    y1 = Dropout(dopara)(y)
    y = Dense(nwidth, activation='relu', name='hidden01')(y1)
    y = BatchNormalization()(y)
    y2 = Dropout(dopara)(y)
    y = Dense(nwidth, activation='relu', name='hidden02')(y2)
    y = BatchNormalization()(y)
    y3 = Dropout(dopara)(y)
    y = Dense(nwidth, activation='relu', name='hidden03')(y3)
    y = BatchNormalization()(y)
    y4 = Dropout(dopara)(y)
    x = concatenate([x,y1,y2,y3,y4],axis=1)
    x = Dense(nwidth, activation='sigmoid', name='hidden6')(x)
    x = Dropout(dopara)(x)
    x = Dense(1024, activation='sigmoid', name='hidden7')(x)
    x = Dropout(dopara)(x)
    
    # Here 3 tasks with 6 outputs
    out1 = Dense(2, activation='softmax', name='output0')(x)
    out2 = Dense(2, activation='softmax', name='output1')(x)
    out3 = Dense(2, activation='softmax', name='output2')(x)
    out = concatenate([out1,out2,out3],axis=1)
#    out = Dense(numofc, activation='sigmoid', name='output')(x)
    
    model = Model(input_img,out)
    model.summary()
    return model

if __name__ == '__main__':
    X = np.zeros((10,1,15,1))
    dnn_mt_en(X)
    dwmlp_mt_en(X)
    dwmlp_st(X)
    