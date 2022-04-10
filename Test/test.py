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
import glob

import seaborn as sns
from math import exp
from subprocess import CalledProcessError, check_call
import traceback
from random import shuffle, random, seed, sample
from numpy import newaxis
import matplotlib.pyplot as plt
import time

from numpy import asarray
from sklearn.preprocessing import OneHotEncoder

import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

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

samples_test = listdir(map_dir)

#model = load_model(path.join('../Examples', 'MODELS', 'Dockground', '0_model'))
model = load_model(path.join('../Examples', 'MODELS', 'BM5', '0_model'))

predictions_file = open('../Examples/predictions_SCR', 'w')

def load_map(sample_path):
    check_call(
        [
            'lz4', '-d', '-f',
            sample_path
        ],
        stdout=sys.stdout)
    X_train, y_train, reg_type, res_pos,_,info = load_obj(sample_path.replace('.pkl.lz4',''))
    remove(sample_path.replace('.lz4',''))
    return X_train, y_train, reg_type, res_pos, info

fold = 'test'
batch_samples_test_1 = []
batch_samples_test_0 = []
for pair in samples_test:
    batch_samples_test_1 += glob.glob(path.join(map_dir,pair,'1','*'))
    batch_samples_test_0 += glob.glob(path.join(map_dir,pair,'0','*'))
batch_samples_test = batch_samples_test_1 + batch_samples_test_0

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
onehot = encoder.fit(np.asarray([['S'], ['C'], ['R']]))


predictions_file.write('Conf' + '\t' +
                       'Fold' + '\t' +
                       'Scores' + '\t' +
                       'Regions' + '\t' +
                       'Score' + '\t' +
                       'Time' + '\t' +
                       'Class' + '\t' +
                       'RecLig' + '\t' +
                       'ResNumber' + '\n')

for test_interface in batch_samples_test:
    try:
        print(test_interface)
        X_test, y_test, reg_type, res_pos, info = load_map(test_interface)
        X_aux = encoder.transform(list(map(lambda x: [x], reg_type)))
        if len(X_test) == 0 or len(X_aux) != len(X_test):
            continue
    except Exception as e:
        logging.info("Bad target complex!" + '\nError message: ' + str(e) + 
                     "\nMore information:\n" + traceback.format_exc())
        continue
        
    start = time.time()
    all_scores = model.predict([X_test, X_aux], batch_size=X_test.shape[0])
    end = time.time()
    _ = gc.collect()

    test_preds = all_scores.mean()
    print(y_test)
    predictions_file.write(test_interface + '\t' +
                        str(fold) + '\t' +
                        ','.join(list(map(lambda x: str(x[0]), all_scores))) + '\t' +
                        ','.join(reg_type) + '\t' +
                        str(test_preds) + '\t' +
                        str(end-start) + '\t' +
                        str(y_test[0]) + '\t' +
                        ','.join(list(map(lambda x: x[3], info))) + '\t' +
                        ','.join(list(map(lambda x: str(x[2]), info))) + '\n')
predictions_file.close()
