#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 21:52:52 2018

@author: xinliang
"""

import numpy as np
import cv2

def playvideo(vd,y,startFrame=0,endFrame=vd.shape[0]):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,25)
    fontScale              = 0.5
    fontColor              = (255,0,255)
    lineType               = 2
    
    bottomLeftCornerOfTextY = (10,45)
    fontColorY              = (0,255,255)
    
    ret=vd.shape[0]
    fps=25
    mspf=int(200/fps)
    w=vd.shape[2]
    h=vd.shape[1]
    frame=np.zeros([w,h,3])
    for f in range(startFrame,endFrame):
        frame=np.copy(vd[f,:,:,:])
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        cv2.putText(frame,'frame'+str(f+1), 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)
        cv2.putText(frame,'groundtruth:'+label[y[f]], 
        bottomLeftCornerOfTextY, 
        font, 
        fontScale,
        fontColorY,
        lineType)
        cv2.imshow('frame',frame)
        if cv2.waitKey(mspf) & 0xFF == ord('q'):
            break
    
    #cap.release()
    cv2.destroyAllWindows()