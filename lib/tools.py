#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 16:33:47 2020

@author: mohseni
"""

import logging
import numpy as np
import pickle
import shutil
import pypdb
import pandas as pd
from prody import *
from os import path, mkdir, remove, getenv, listdir, system
from io import StringIO
import urllib
import re
import glob
from subprocess import CalledProcessError, check_call
import traceback
import sys
import gzip

#========================================================
#NACCESS
NACCESS_PATH='naccess'
#========================================================


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def do_processing(cases, function, use_multiprocessing):
    if use_multiprocessing:
        import multiprocessing
        max_cpus = 30
        manager = multiprocessing.Manager()
        report_dict = manager.dict()
        pool = multiprocessing.Pool(processes = min(max_cpus, multiprocessing.cpu_count()))
    else:
        report_dict = dict()

    for args in cases:
        args += (report_dict,)
        if use_multiprocessing:
            pool.apply_async(function, args = args)
        else:
            function(*args)

    if use_multiprocessing:
        pool.close()
        pool.join()
    
    return dict(report_dict)
