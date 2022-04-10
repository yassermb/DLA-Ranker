#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:09:56 2020

@author: yasser
"""

import logging
import gc
import os
import sys
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
from scipy import interp

import collections
#import scr
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder

import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential,load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, AveragePooling3D, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Layer, BatchNormalization, Add, Lambda
import tensorflow as tf
from tensorflow.keras.constraints import max_norm

from tensorflow.keras.layers import Dot
from tensorflow.keras.backend import ones, ones_like
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight

import pickle

from sklearn.preprocessing import OneHotEncoder

def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

NB_EPOCH = 20
hidden_size1 = 200
hidden_size2 = 20
v_dim = 24

atom_channels = 167
#atom_channels = 4

logging.basicConfig(filename='manager.log', filemode='w', format='%(levelname)s: %(message)s', level=logging.DEBUG)
mainlog = logging.getLogger('main')
logging.Logger

seed(int(np.round(np.random.random()*10)))

map_dir_dock = '../Examples/map_dir'
confom_dict = pd.read_csv('../Examples/conformations_list.txt', sep=';')   

print('Your python version: {}'.format(sys.version_info.major))

USE_TENSORFLOW_AS_BACKEND = True

FORCE_CPU = False

if USE_TENSORFLOW_AS_BACKEND:
    os.environ['KERAS_BACKEND'] = 'tensorflow'
else:
    os.environ['KERAS_BACKEND'] = 'theano'
if FORCE_CPU:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
if USE_TENSORFLOW_AS_BACKEND == True:
    import tensorflow
    print('Your tensorflow version: {}'.format(tensorflow.__version__))
    print("GPU : "+tensorflow.test.gpu_device_name())
else:
    import theano
    print('Your theano version: {}'.format(theano.__version__))    
    
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
onehot = encoder.fit(np.asarray([['S'], ['C'], ['R']]))


def Conv_3D_model(input_shape, input_shape_aux):
    X_in = Input(shape=input_shape)
    aux_input = Input(shape=input_shape_aux)

    H = Conv3D(20, kernel_size=(1, 1, 1), use_bias = True, padding = 'valid', activation='linear', kernel_initializer='he_uniform', input_shape=X_in.shape)(X_in)
    H = BatchNormalization()(H)    
    H = Conv3D(20, kernel_size=(3, 3, 3), use_bias = True, padding = 'valid', activation='elu', kernel_initializer='he_uniform', input_shape=H.shape)(H)
    H = BatchNormalization()(H)
    H = Conv3D(30, kernel_size=(4, 4, 4), use_bias = True, padding = 'valid', activation='elu', kernel_initializer='he_uniform', input_shape=H.shape)(H)
    H = BatchNormalization()(H)
    H = Conv3D(20, kernel_size=(4, 4, 4), use_bias = True, padding = 'valid', activation='elu', kernel_initializer='he_uniform', input_shape=H.shape)(H)
    H = BatchNormalization()(H)
    H = AveragePooling3D(pool_size=(4, 4, 4), strides=(4, 4, 4))(H)
    H = Flatten()(H)
    H = Dropout(0.4)(H)
    
    H = Concatenate()([H, aux_input])
    
    H = Dense(hidden_size1, activation='elu', name='layer1', kernel_constraint=max_norm(4), bias_constraint=max_norm(4))(H)
    H = Dropout(0.2)(H)
    
    H = Dense(hidden_size2, activation='elu', name='layer2', kernel_constraint=max_norm(4), bias_constraint=max_norm(4))(H)
    H = Dropout(0.1)(H)
    
    Y = Dense(1, activation='sigmoid')(H)

    _model = Model(inputs=[X_in, aux_input], outputs=Y)
    _model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
    _model.summary()
    return _model


def Conv_3D_model_4channels(input_shape, input_shape_aux):
    X_in = Input(shape=input_shape)
    aux_input = Input(shape=input_shape_aux)
    
    H = Conv3D(10, kernel_size=(3, 3, 3), use_bias = True, padding = 'valid', activation='elu', kernel_initializer='he_uniform', input_shape=X_in.shape)(X_in)
    H = BatchNormalization()(H)
    H = Conv3D(10, kernel_size=(4, 4, 4), use_bias = True, padding = 'valid', activation='elu', kernel_initializer='he_uniform', input_shape=H.shape)(H)
    H = BatchNormalization()(H)
    H = Conv3D(10, kernel_size=(4, 4, 4), use_bias = True, padding = 'valid', activation='elu', kernel_initializer='he_uniform', input_shape=H.shape)(H)
    H = BatchNormalization()(H)
    H = AveragePooling3D(pool_size=(4, 4, 4), strides=(4, 4, 4))(H)
    H = Flatten()(H)
    H = Dropout(0.4)(H)
    
    H = Concatenate()([H, aux_input])
    
    H = Dense(hidden_size1, activation='elu', name='layer1', kernel_constraint=max_norm(4), bias_constraint=max_norm(4))(H)
    H = Dropout(0.2)(H)
    
    H = Dense(hidden_size2, activation='elu', name='layer2', kernel_constraint=max_norm(4), bias_constraint=max_norm(4))(H)
    H = Dropout(0.1)(H)
    
    Y = Dense(1, activation='sigmoid')(H)

    _model = Model(inputs=[X_in, aux_input], outputs=Y)
    _model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001))
    _model.summary()
    return _model

def load_map(sample_path):
    check_call(
        [
            'lz4', '-d', '-f',
            sample_path
        ],
        stdout=sys.stdout)
    X_train, y_train, reg_type, _,_,_ = load_obj(sample_path.replace('.pkl.lz4',''))
    remove(sample_path.replace('.lz4',''))
    return X_train, y_train, reg_type

tprs = []
aucs = []
base_fpr = np.linspace(0, 1, 101)
plt.figure(figsize=(10,10))


samples_train = glob.glob(path.join(map_dir_dock,'*','*','*'))
shuffle(samples_train)
samples_train = samples_train[:int(0.7*len(samples_train))]
samples_test = samples_train[:int(0.7*len(samples_train))]
    
fhandler_train = open('train_log', 'w')
fhandler_test = open('test_log', 'w')


class_weights = class_weight.compute_class_weight('balanced', np.unique(confom_dict.Class.to_list()), confom_dict.Class.to_list())
d_class_weights = dict(enumerate(class_weights))


for foldk in ['Total']:
    seed(int(np.round(np.random.random()*10)))
    
    input_shape=(v_dim,v_dim,v_dim,atom_channels+6)
    
    if atom_channels == 4:
        model  = Conv_3D_model_4channels(input_shape, 3)
    else:
        model  = Conv_3D_model(input_shape, 3)
    
    #model = load_model('Total_0_model')
            
    with open(str(foldk) + '_train_interfaces.txt', 'w') as f_handler_trainlist:
        for inter in samples_train:
            f_handler_trainlist.write(inter+'\n')
    with open(str(foldk) + '_test_interfaces.txt', 'w') as f_handler_testlist:
        for inter in samples_test:
            f_handler_testlist.write(inter+'\n')
    
    step = 40
    for epoch in range(11, NB_EPOCH+1):
        for batch_i in range(0, len(samples_train), step):
            try:
                t = time.time()
                X_train, y_train, X_train_aux = None, None, None
                for step_i in range(step):
                    X_train_tmp, y_train_tmp, reg_type_tmp = load_map(samples_train[batch_i+step_i])
                    X_train_aux_tmp = encoder.transform(list(map(lambda x: [x], reg_type_tmp)))
                    if len(X_train_tmp)==0:
                        continue
                    if len(X_train_aux_tmp) != len(X_train_tmp):
                        continue
                    
                    if step_i == 0:
                        X_train, y_train, X_train_aux = X_train_tmp, y_train_tmp, X_train_aux_tmp
                        continue
                    
                    
                    X_train = np.concatenate((X_train, X_train_tmp))
                    y_train = np.concatenate((y_train, y_train_tmp))
                    X_train_aux = np.concatenate((X_train_aux, X_train_aux_tmp))
                
                if len(X_train_aux) != len(X_train):
                    raise Exception()    
                
                X_val = np.array([X_train[0]])
                y_val = np.array([y_train[0]])
                X_val_aux = np.array([X_train_aux[0]])
            except:
                print("pass")
                continue
                
            if X_train.shape[0] > 2800:
                X_train = X_train[:2800]
                y_train = y_train[:2800]
                X_train_aux = X_train_aux[:2800]
            
            X_train_input = [X_train, X_train_aux] 
            model.fit(X_train_input, y_train, batch_size=X_train.shape[0], epochs=1, verbose = 0, class_weight=d_class_weights)
            
            start = time.time()
            train_preds = model.predict(X_train_input, batch_size=X_train.shape[0])
            end = time.time()
            
            X_val_input = [X_val, X_val_aux]
            val_preds = model.predict(X_val_input, batch_size=X_val.shape[0])
            
            _ = gc.collect()
            
            fhandler_train.write('NA' + '\t' +
                                 str(foldk)  + '\t' +
                                 str(epoch)  + '\t' +
                                 ','.join(list(map(lambda x: str(x[0]), train_preds))) + '\t' +
                                 'NA' + '\t' +
                                    str(train_preds.mean()) + '\t' +
                                    str(end-start) + '\t' +
                                 ','.join(list(map(lambda x: str(x), y_train))) + '\n')
            
            print('\n')
            print(np.array(y_train).flatten())
            print('\n')
            print(train_preds.flatten())
            print('\n')
            print(np.array(y_val).flatten())
            print('\n')
            print(val_preds.flatten())
            print('\n')
        
            # Train / validation scores
            train_acc = accuracy_score(np.round(train_preds), y_train)
            val_acc = accuracy_score(np.round(val_preds), y_val)
            
            train_val_loss = [train_acc, val_acc]
                        
            print("Fold: {}".format(foldk),
                  "Epoch: {:04d}".format(epoch),
                  "Batch index: {:04d}".format(batch_i),
                  "train_acc= {:.4f}".format(train_val_loss[0]),
                  "val_acc= {:.4f}".format(train_val_loss[1]),
                  "time= {:.4f}".format(time.time() - t))
        
        model.save(str(foldk)+'_'+str(epoch)+'_model')
            
        test_preds = []
        Xts_lbl = []
        for test_interface in samples_test:
            try:
                X_test, y_test, reg_type = load_map(test_interface)
                X_aux = encoder.transform(list(map(lambda x: [x], reg_type)))
                if len(X_test) == 0 or len(X_aux) != len(X_test):
                    continue
            except:
                continue
        
            X_test_input = [X_test, X_aux]
            start = time.time()
            all_scores = model.predict(X_test_input, batch_size=X_test.shape[0])
            end = time.time()
            _ = gc.collect()


            test_preds.append(all_scores.mean())
            Xts_lbl.append(y_test[0])
            fhandler_test.write(test_interface + '\t' +
                                str(foldk) + '\t' +
                                str(epoch)  + '\t' +
                                ','.join(list(map(lambda x: str(x[0]), all_scores))) + '\t' +
                                ','.join(reg_type) + '\t' +
                                str(test_preds[-1]) + '\t' +
                                str(end-start) + '\t' +
                                str(y_test[0]) + '\n')
            
        noskill = [0]*len(Xts_lbl)
            
        
        np.save('Xts_lbl_'+str(foldk)+'_'+str(epoch), Xts_lbl)
        np.save('test_preds_'+str(foldk)+"_"+str(epoch), test_preds)
        np.save('noskill_'+str(epoch), noskill)
        print('Xts_lbl: ', Xts_lbl)
        print('test_preds: ', test_preds)
        auc = roc_auc_score(Xts_lbl, test_preds)
        print('Logistic: ROC AUC=%.3f' % (auc))