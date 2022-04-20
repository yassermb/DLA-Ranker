#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 20:21:06 2022

@author: yasser
"""

import logging
import os
import sys
from os import path, mkdir, getenv, listdir, remove, system, stat
import pandas as pd
import numpy as np
from prody import *
import glob
import shutil
#import matplotlib.pyplot as plt
import seaborn as sns
from math import exp
from subprocess import CalledProcessError, check_call, call
import traceback
from random import shuffle, random, seed, sample
from numpy import newaxis
import matplotlib.pyplot as plt
import time
from prody import *
import collections
import scr
from numpy import asarray
from sklearn.preprocessing import OneHotEncoder
import subprocess
import load_data as load
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(filename='manager.log', filemode='w', format='%(levelname)s: %(message)s', level=logging.DEBUG)
mainlog = logging.getLogger('main')
logging.Logger

sys.path.insert(1, '../lib/')
import tools as tl

confom_dict = pd.read_csv('../Examples/conformations_list.txt', sep=';')    
comp_dir = '../Examples/conformations_directory'
target_comp = listdir(comp_dir)

map_dir = '../Examples/map_dir'
if not path.exists(map_dir):
    mkdir(map_dir)


bin_path = "./maps_generator"
v_dim = 24

def mapcomplex(file, pose_class, ch1, ch2, pair, pose):
    try:
        name = pair+'_'+str(pose)
        
        rec = parsePDB(file).select('protein').select('chain ' + ch1[0])
        rec.setChids('R')
        lig = parsePDB(file).select('protein').select('chain ' + ch2[0])
        lig.setChids('L')    
        
        writePDB(name+'_r.pdb', rec.toAtomGroup())
        writePDB(name+'_l.pdb', lig.toAtomGroup())
        writePDB(name+'_complex.pdb', rec.toAtomGroup() + lig.toAtomGroup())
        
        scr.get_scr(name+'_r.pdb', name+'_l.pdb', name+'_complex.pdb', name)
        
        rimcoresup = pd.read_csv(name+'_rimcoresup.csv', header=None, sep=' ')
        rec_regions = rimcoresup.loc[rimcoresup[4] == 'receptor']
        rec_regions = pd.Series(rec_regions[5].values, index=rec_regions[2]).to_dict()
        lig_regions = rimcoresup.loc[rimcoresup[4] == 'ligand']
        lig_regions = pd.Series(lig_regions[5].values, index=lig_regions[2]).to_dict()
        
        res_num2name_map_rec = dict(zip(rec.getResnums(),rec.getResnames()))
        res_num2name_map_lig = dict(zip(lig.getResnums(),lig.getResnames()))
        res_num2coord_map_rec = dict(zip(rec.select('ca').getResnums(),rec.select('ca').getCoords()))
        res_num2coord_map_lig = dict(zip(lig.select('ca').getResnums(),lig.select('ca').getCoords()))
        
        L1 = list(set(rec.getResnums()))
        res_ind_map_rec = dict([(x,inx) for inx, x in enumerate(sorted(L1))])
        L1 = list(set(lig.getResnums()))
        res_ind_map_lig = dict([(x,inx+len(res_ind_map_rec)) for inx, x in enumerate(sorted(L1))])
        
        res_inter_rec = [(res_ind_map_rec[x], rec_regions[x], x, 'R', res_num2name_map_rec[x], res_num2coord_map_rec[x]) 
                          for x in sorted(list(rec_regions.keys())) if x in res_ind_map_rec]
        res_inter_lig = [(res_ind_map_lig[x], lig_regions[x], x, 'L', res_num2name_map_lig[x], res_num2coord_map_lig[x])
                          for x in sorted(list(lig_regions.keys())) if x in res_ind_map_lig]
        
        reg_type =  list(map(lambda x: x[1],res_inter_rec)) + list(map(lambda x: x[1],res_inter_lig))
        res_name =  list(map(lambda x: [x[4]],res_inter_rec)) + list(map(lambda x: [x[4]],res_inter_lig))
        res_pos =  list(map(lambda x: x[5],res_inter_rec)) + list(map(lambda x: x[5],res_inter_lig))


        #Merge these two files!
        with open('resinfo','w') as fh_res:
            for x in res_inter_rec:
                fh_res.write(str(x[2])+';'+x[3]+'\n')
            for x in res_inter_lig:
                fh_res.write(str(x[2])+';'+x[3]+'\n')  

        with open('scrinfo','w') as fh_csr:
            for x in res_inter_rec:
                fh_csr.write(str(x[2])+';'+x[3]+';'+x[1]+'\n')
            for x in res_inter_lig:
                fh_csr.write(str(x[2])+';'+x[3]+';'+x[1]+'\n')
    
        if not res_inter_rec or not res_inter_lig:
            return [],[],[]
        
        #tl.coarse_grain_pdb('train.pdb')
        mapcommand = [bin_path, "--mode", "map", "-i", name+'_complex.pdb', "--native", "-m", str(v_dim), "-t", "167", "-v", "0.8", "-o", name+'_complex.bin']
        call(mapcommand)
        dataset_train = load.read_data_set(name+'_complex.bin')
        
        print(dataset_train.maps.shape)
        
        #scaler = MinMaxScaler()
        #scaler.fit(dataset_train.maps)
        #data_norm = scaler.transform(dataset_train.maps)
        data_norm = dataset_train.maps
        
        X = np.reshape(data_norm, (-1,v_dim,v_dim,v_dim,173))
        y = [int(pose_class)]*(len(res_inter_rec) + len(res_inter_lig))
        
        map_name = path.join(map_dir, pair, pose_class, name)
        tl.save_obj((X,y,reg_type,res_pos,res_name,res_inter_rec+res_inter_lig), map_name)
        
        check_call(
            [
                'lz4', '-f',   #, '--rm' because if inconsistency in lz4 versions! 
                map_name + '.pkl'
            ],
            stdout=sys.stdout)
        remove(map_name + '.pkl')
        remove(name+'_complex.bin')
        remove(name+'_r.pdb')
        remove(name+'_l.pdb')
        remove(name+'_complex.pdb')
        remove(name+'_rimcoresup.csv')
        
        
        print(type(X))
        print(X.shape)
        
        
    except Exception as e:
        logging.info("Bad interface!" + '\nError message: ' + str(e) + 
                      "\nMore information:\n" + traceback.format_exc())
        return [],[],[]
    
    return X, y, reg_type


already_exist = listdir(map_dir)
def process_targetcomplex(targetcomplex, comp_dir, report_dict):
    try:
        if targetcomplex in already_exist:
            return
        logging.info('Processing ' + targetcomplex + ' ...')
        pos_path = path.join(map_dir, targetcomplex, '1')
        neg_path = path.join(map_dir, targetcomplex, '0')
        if not path.exists(path.join(map_dir, targetcomplex)):
            mkdir(path.join(map_dir, targetcomplex))
        if not path.exists(pos_path):
            mkdir(pos_path)    
        if not path.exists(neg_path):
            mkdir(neg_path)

        good_poses = confom_dict.loc[(confom_dict.Comp == targetcomplex) & (confom_dict.Class == 1)].Conf.to_list()
        bad_poses = confom_dict.loc[(confom_dict.Comp == targetcomplex) & (confom_dict.Class == 0)].Conf.to_list()
        ch1, ch2 =  confom_dict.loc[confom_dict.Comp == targetcomplex][['ch1','ch2']].iloc[0]
        for pose in good_poses:
            file = path.join(comp_dir, targetcomplex, pose + '.pdb')
            mapcomplex(file, '1', ch1, ch2, targetcomplex, path.basename(pose))
        for pose in bad_poses:
            file = path.join(comp_dir, targetcomplex, pose + '.pdb')
            mapcomplex(file, '0', ch1, ch2, targetcomplex, path.basename(pose))
    
    except Exception as e:
        logging.info("Bad target complex!" + '\nError message: ' + str(e) + 
                      "\nMore information:\n" + traceback.format_exc())

def manage_pair_files(use_multiprocessing):
    tc_cases = []
    for tc in target_comp:
        tc_cases.append((tc, comp_dir))
    report_dict = tl.do_processing(tc_cases, process_targetcomplex, use_multiprocessing)
    return report_dict

report_dict = manage_pair_files(False)