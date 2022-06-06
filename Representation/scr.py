#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 23:49:27 2021

@author: yasser
"""

import re
import os
import sys
from os import path, remove
import glob

sys.path.insert(1, '../lib/')
import tools as tl

def rimcoresup(rsa_rec,rsa_lig,rsa_complex):
    '''INPUT: file rsa da NACCESS.

       ###rASAm: relative ASA in monomer
       ###rASAc: relative ASA in complex

       ###Levy model
       ###deltarASA=rASAm-rASAc 
       ###RIM      deltarASA > 0 and rASAc >= 25 and rASAm >= 25   ###corretto da rASAc > 25 and rASAm > 25
       ###CORE     deltarASA > 0 and rASAm >= 25 and rASAc < 25    ###corretto da rASAm > 25 and rASAc <= 25
       ###SUPPORT  deltarASA > 0 and rASAc < 25 and rASAm < 25

       OUTPUT:rim, core, support'''

    ASA1=[]
    resNUMasa1=0
    lines = [line.rstrip('\n') for line in open(rsa_rec)]
    for lineee in lines:
        a=re.split(' ',lineee)
        a=list(filter(None, a))
        if a[0] == 'RES' and len(a) == 14:
            resNUMasa1=(resNUMasa1)+1
            restype=a[1]
            chain=a[2]
            resnumb=a[3]
            resnumb=re.findall('\d+', resnumb)   
            resnumb=resnumb[0]
            rASAm=a[5]
            ASA1.append((restype,chain,int(resnumb),rASAm,'receptor'))
        elif a[0] == 'RES' and len(a) == 13:
            resNUMasa1=(resNUMasa1)+1
            restype=a[1]
            testchain=re.findall('\d+|\D+', a[2])
            if len(testchain) == 1:
                chain=''
                resnumb=int(testchain[0])
            elif len(testchain) == 2:
                primoterm=testchain[0]
                if primoterm.isdigit():
                    chain=''
                    resnumb=int(testchain[0])
                else:
                    chain=testchain[0]
                    resnumb=int(testchain[1])           
            #resnumb=re.findall('\d+', resnumb)
            #resnumb=resnumb[0]
            rASAm=a[4]
            ASA1.append((restype,chain,int(resnumb),rASAm,'receptor'))

            
    ASA2=[]
    resNUMasa2=0
    lines = [line.rstrip('\n') for line in open(rsa_lig)]
    for lineee in lines:
        a=re.split(' ',lineee)
        a=list(filter(None, a))
        if a[0] == 'RES' and len(a) == 14:
            resNUMasa2=(resNUMasa2)+1
            restype=a[1]
            chain=a[2]
            resnumb=a[3]
            resnumb=re.findall('\d+', resnumb)
            resnumb=resnumb[0]
            rASAm=a[5]
            ASA2.append((restype,chain,int(resnumb),rASAm,'ligand'))
        elif a[0] == 'RES' and len(a) == 13:
            resNUMasa2=(resNUMasa2)+1
            restype=a[1]
            testchain=re.findall('\d+|\D+', a[2])
            if len(testchain) == 1:
                chain=''
                resnumb=int(testchain[0])
            elif len(testchain) == 2:
                primoterm=testchain[0]
                if primoterm.isdigit():
                    chain=''
                    resnumb=int(testchain[0])
                else:
                    chain=testchain[0]
                    resnumb=int(testchain[1])              
            #resnumb=re.findall('\d+', resnumb)
            #resnumb=resnumb[0]
            rASAm=a[4]
            ASA2.append((restype,chain,int(resnumb),rASAm,'ligand'))

            
    ASAfull=[]
    resNUMasafull=0
    lines = [line.rstrip('\n') for line in open(rsa_complex)]
    for lineee in lines:
        a=re.split(' ',lineee)
        a=list(filter(None, a))
        if a[0] == 'RES' and len(a) == 14:
            resNUMasafull=resNUMasafull+1
            restype=a[1]
            chain=a[2]
            resnumb=a[3]
            resnumb=re.findall('\d+', resnumb)
            resnumb=resnumb[0]
            rASAm=a[5]
            if resNUMasafull <= len(ASA1):
                filename='receptor'
            else:
                filename='ligand'
            ASAfull.append((restype,chain,int(resnumb),rASAm,filename))
        elif a[0] == 'RES' and len(a) == 13:
            resNUMasafull=resNUMasafull+1
            restype=a[1]
            testchain=re.findall('\d+|\D+', a[2])
            if len(testchain) == 1:
                chain=''
                resnumb=int(testchain[0])
            elif len(testchain) == 2:
                primoterm=testchain[0]
                if primoterm.isdigit():
                    chain=''
                    resnumb=int(testchain[0])
                else:
                    chain=testchain[0]
                    resnumb=int(testchain[1])              
            #resnumb=re.findall('\d+', resnumb)
            #resnumb=resnumb[0]
            rASAm=a[4]
            if resNUMasafull <= len(ASA1):
                filename='receptor'
            else:
                filename='ligand'
            ASAfull.append((restype,chain,int(resnumb),rASAm,filename))

    rim=[]
    core=[]
    support=[]
    for elements in ASAfull:
        for x in ASA1:
            if elements[0:3] == x[0:3] and elements[4] == x[4]:
                rASAm=float(x[3])
                rASAc=float(elements[3])
                deltarASA=rASAm-rASAc
                if deltarASA > 0:
                    if rASAm < 25:# and rASAc < 25:
                        support.append(x)
                    elif rASAm > 25:
                        if rASAc <= 25:
                            core.append(x)
                        else:
                            rim.append(x)
                                    
        for x in ASA2:        
            if elements[0:3] == x[0:3] and elements[4] == x[4]:
                rASAm=float(x[3])
                rASAc=float(elements[3])
                deltarASA=rASAm-rASAc
                if deltarASA > 0:
                    if rASAm < 25:# and rASAc < 25:
                        support.append(x)
                    elif rASAm > 25:
                        if rASAc <= 25:
                            core.append(x)
                        else:
                            rim.append(x)

    return rim, core, support

def get_scr(rec, lig, com, name):
    
    if tl.USE_FREESASA:
        cmdcompl=tl.NACCESS_PATH + ' --format=rsa ' + com + ' > ' + path.basename(com.replace('pdb', 'rsa'))
        os.system(cmdcompl) 
        cmdrec=tl.NACCESS_PATH + ' --format=rsa ' + rec + ' > ' + path.basename(rec.replace('pdb', 'rsa'))
        os.system(cmdrec) 
        cmdlig=tl.NACCESS_PATH + ' --format=rsa ' + lig + ' > ' + path.basename(lig.replace('pdb', 'rsa'))
        os.system(cmdlig)    
    else:
        cmdcompl=tl.NACCESS_PATH + ' ' + com
        os.system(cmdcompl) 
        cmdrec=tl.NACCESS_PATH + ' ' + rec
        os.system(cmdrec) 
        cmdlig=tl.NACCESS_PATH + ' ' + lig
        os.system(cmdlig)
 
    # ('GLN', 'B', '44', '55.7', 'receptor')
    rim,core,support = rimcoresup(path.basename(rec.replace('pdb', 'rsa')), path.basename(lig.replace('pdb', 'rsa')), path.basename(com.replace('pdb', 'rsa')))
    outprimcoresup = open(name+'_rimcoresup.csv', 'w')

    for elementrim in rim:
        outprimcoresup.write(str((' '.join(map(str,elementrim)))+" R")+"\n") #Rim
    for elementcore in core:
        outprimcoresup.write(str((' '.join(map(str,elementcore)))+" C")+"\n") #Core
    for elementsup in support:
        outprimcoresup.write(str((' '.join(map(str,elementsup)))+" S")+"\n") #Support

    outprimcoresup.close()
    
    for f in glob.glob("*.rsa"):
        try:
            remove(f)
        except:
            continue
        
    for f in glob.glob("*.asa"):
        try:
            remove(f)
        except:
            continue
    
    """
    remove(path.basename(rec.replace('pdb', 'rsa')))    
    remove(path.basename(lig.replace('pdb', 'rsa')))
    remove(path.basename(com.replace('pdb', 'rsa')))
    remove(path.basename(rec.replace('pdb', 'asa')))
    remove(path.basename(lig.replace('pdb', 'asa')))
    remove(path.basename(com.replace('pdb', 'asa')))
    """
    
    try:
        remove(path.basename(rec.replace('pdb', 'log')))
        remove(path.basename(lig.replace('pdb', 'log')))
        remove(path.basename(com.replace('pdb', 'log')))
    except:
        pass
########################