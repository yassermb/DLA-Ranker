#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:09:56 2020

@author: yasser
"""

import logging
import os
import sys
import gc
from os import path, mkdir, getenv, listdir, remove, system, stat
import pandas as pd
import numpy as np
#from prody import *
import glob
import shutil
#import matplotlib.pyplot as plt
import seaborn as sns
from math import exp
import subprocess
from subprocess import CalledProcessError, check_call
import traceback
from random import shuffle, random, seed, sample
from numpy import newaxis
import matplotlib.pyplot as plt
import time

import collections
#import scr
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder

import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.datasets import mnist # subroutines for fetching the MNIST dataset
from tensorflow.keras.models import Model, Sequential,load_model # basic class for specifying and training a neural network
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, AveragePooling3D
#from tensorflow.keras.utils import np_utils # utilities for one-hot encoding of ground truth values

from tensorflow.keras.layers import Dot
from tensorflow.keras.backend import ones, ones_like
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import pickle

print('Your python version: {}'.format(sys.version_info.major))
USE_TENSORFLOW_AS_BACKEND = True
# IF YOU *DO* HAVE AN Nvidia GPU on your computer, or execute on Google COLAB, then change below to False!
FORCE_CPU = False #False 
if USE_TENSORFLOW_AS_BACKEND:
    os.environ['KERAS_BACKEND'] = 'tensorflow'
else:
    os.environ['KERAS_BACKEND'] = 'theano'
if FORCE_CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
if USE_TENSORFLOW_AS_BACKEND == True:
    import tensorflow as tf
    print('Your tensorflow version: {}'.format(tf.__version__))
    print("GPU : "+tf.test.gpu_device_name())
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    import theano
    print('Your theano version: {}'.format(theano.__version__))

logging.basicConfig(filename='manager.log', filemode='w', format='%(levelname)s: %(message)s', level=logging.DEBUG)
mainlog = logging.getLogger('main')
logging.Logger

seed(int(np.round(np.random.random()*10)))
#################################################################################################

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

v_dim = 24

map_dir = '../Examples/map_dir'
intermediate_dir = '../Examples/intermediate'

samples_test = listdir(map_dir)

if path.isdir(intermediate_dir):
    shutil.rmtree(intermediate_dir)
mkdir(intermediate_dir)

model = load_model(path.join('../Examples', 'MODELS', 'Dockground', '0_model'))

def load_map(sample_path):
    check_call(
        [
            'lz4', '-d', '-f',
            sample_path
        ],
        stdout=sys.stdout)
    X_train, y_train, reg_type, res_pos,_,_ = load_obj(sample_path.replace('.pkl.lz4',''))
    remove(sample_path.replace('.lz4',''))
    return X_train, y_train, reg_type, res_pos
        
batch_samples_test_1 = []
batch_samples_test_0 = []
for pair in samples_test:
    batch_samples_test_1 += glob.glob(path.join(map_dir,pair,'1','*'))
    batch_samples_test_0 += glob.glob(path.join(map_dir,pair,'0','*'))
    mkdir(path.join(intermediate_dir, pair))
    mkdir(path.join(intermediate_dir, pair, '1'))
    mkdir(path.join(intermediate_dir, pair, '0'))

batch_samples_test = batch_samples_test_1 + batch_samples_test_0

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
onehot = encoder.fit(np.asarray([['S'], ['C'], ['R']]))

for test_interface in batch_samples_test:
    try:
        print(test_interface)
        X_test, y_test, reg_type, res_pos = load_map(test_interface)
        X_aux = encoder.transform(list(map(lambda x: [x], reg_type)))
        if len(X_test) == 0 or len(X_aux) != len(X_test):
            continue
    except Exception as e:
        logging.info("Bad interface!" + '\nError message: ' + str(e) + 
                      "\nMore information:\n" + traceback.format_exc())
        continue
        
    intermediate_model = Model(inputs=model.input, outputs=model.get_layer('layer1').output)
    intermediate_prediction = intermediate_model.predict([X_test, X_aux], batch_size=X_test.shape[0])
    _ = gc.collect()

    with open(test_interface.replace(map_dir, intermediate_dir).replace('.pkl','.graph'),'w') as f_handler_graph:
        for i in range(len(X_test)):
            f_handler_graph.write(','.join(list(map(lambda x: str(x), res_pos[i]))) + ',' +
                                  reg_type[i] + ',' +
                                  ','.join(list(map(lambda x: str(x), intermediate_prediction[i]))) + '\n')