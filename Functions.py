import cv2, imutils
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage.util import random_noise as imnoise
from scipy import signal,ndimage

def AddGaussianNoise(pin, sd, mn, vr):
    hsv = cv2.cvtColor(pin,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    vout = imnoise(v, mode='gaussian', seed=sd, mean = mn, var=vr)
    v = vout*255
    v = v.astype(np.uint8)
    final_hsv = cv2.merge((h,s,v))
    pout = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    return pout

def GaussianFilter(pin, sigma):
        if (sigma>0):
                gmask = GaussianMask(sigma,math.ceil(2*sigma)+1, math.ceil(2*sigma)+1)
                #pout = filter2(pin,gmask)
                pout = cv2.filter2D(pin,-1,gmask)
        else:
                pout = pin   
        return pout

def GaussianMask(sigma,nrow,ncol):
        g = np.zeros((nrow, ncol))
        rcenter = math.floor(nrow/2)+1
        ccenter = math.floor(ncol/2)+1
        pi = np.arctan(1)*4
        s = 0
        for i in range(1,nrow):
                for j in range(1,ncol):
                        g[i][j] = math.exp(-(pow(i-rcenter,2)+pow(j-ccenter,2))/(2*pow(sigma,2)))/(2*pi*pow(sigma,2))
                        s = s + g[i][j]
        g = g/s
        return g