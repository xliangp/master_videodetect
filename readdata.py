#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 16:03:31 2018

@author: xinliang
"""
import struct
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import os
import glob

def read_oneseq(path):
    
    def read_header(ifile):
        feed = ifile.read(4)
        norpix = ifile.read(24)
        version = struct.unpack('@i', ifile.read(4))
        length = struct.unpack('@i', ifile.read(4))
        assert(length != 1024)
        descr = ifile.read(512)
        params = [struct.unpack('@i', ifile.read(4))[0] for i in range(0,9)]
        fps = struct.unpack('@d', ifile.read(8))
        # skipping the rest
        ifile.read(432)
        image_ext = {100:'raw', 102:'jpg',201:'jpg',1:'png',2:'png'}
        return {'w':params[0],'h':params[1],
                'bdepth':params[2],
                'ext':image_ext[params[5]],
                'format':params[5],
                'size':params[4],
                'true_size':params[8],
                'num_frames':params[6],
                'fps':fps}
    
    pathVD=path+'_s.seq'
    ifile = open(pathVD, 'rb')
    params = read_header(ifile)
    bytes = open(pathVD, 'rb').read()      
    
    scale= 0.5
    w=int(params['w']*scale)
    h=int(params['h']*scale) 
    batch_size=224
    
    #images=np.empty([params['num_frames'],h,w,3])   
    x=np.empty([params['num_frames'],batch_size,batch_size,3])
   
    top=params['h']//2-230
    bottom=params['h']//2+230
    left=params['w']//2-300
    right=params['w']//2+300
    
    # this is freaking magic, but it works
    extra = 8
    s = 1024
    seek = [0]*(params['num_frames']+1)
    seek[0] = 1024

    for i in range(0, params['num_frames']-1):
        tmp = struct.unpack_from('@I', bytes[s:s+4])[0]
        s = seek[i] + tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s+1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        seek[i+1] = s
        nbytes = struct.unpack_from('@i', bytes[s:s+4])[0]
        I = bytes[s+4:s+nbytes]
        
        tmp_file = '/tmp/img%d.jpg' % i
        open(tmp_file, 'wb+').write(I)
        
        img = cv2.imread(tmp_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
              
        data = cv2.resize(img[top:bottom,left:right,:], (batch_size,batch_size))
        data=data/255
        x[i,:,:,:]=data
        img = cv2.resize(img, (0,0), fx=scale, fy=scale)
        
        #img=img/255
        #images[i,:,:,:]=img


    '''
    diff_thre=10/255
    mask=np.zeros((h,w))
    
    bg=np.median(images[0:400,:,:,:],axis=0)
    for i in range(params['num_frames']-1):
        cr=images[i,:,:,:]
        mask+=np.sum(abs(cr-bg),axis=2)>diff_thre
        '''
        
    return x,params

def getannotation(path,num_frames,label=[]):
    pathAN= path + '_t.txt'
    num_frames=3975
    y=np.empty([num_frames,],dtype=int)
    dic=(not label)
    with open(pathAN,encoding="latin-1") as f:
        if dic:
            for _ in range(3):
                next(f)
            for _ in range(3,16,1):
                action,_= f.readline().split(' ')
                label.append(action)
            for _ in range(16,19):
                next(f)
        else:
            for _ in range(19):
                next(f)

        for line in f:
            _,s,e,a = line.rstrip('\n').split('    ')
            s=int(s)
            e=int(e)
            for idx,x in enumerate(label):
                if x==a:
                    y[s-1:e]=idx
                    break
                
    return y

def getpossition(path):
    path='validation/train/013009_A29_Block6_C57ma1_t-track.mat'
    mat_contents = sio.loadmat(path)
    position=mat_contents['Y'][0,0,:]
    return position
    
    
vd,x,attributes=read_oneseq('validation/train/013009_A29_Block6_C57ma1')
#bg=np.median(vd)