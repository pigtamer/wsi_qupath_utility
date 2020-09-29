import cv2 as cv
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob

w_tar, h_tar = 256, 256

basepath = "/home/cunyuan/4tb/Kimura/DATA"
dpl = {"IHC": basepath+ "/TILES_(%d, %d)/DAB/*/*/*/" % (size, size),
       "Masks": basepath + "/TILES_(%d, %d)/Masks/*/*/*/" % (size, size)}
       

# dpl = [for x in dpl]

k = 0
l = []
dp = "Masks"

for m in glob.glob(dpl[dp] + "/*.tif"):
    # print(c,d,m,n)
    k+=1
    imm = cv.imread(m).astype(np.uint8)
    imm = cv.cvtColor(imm, cv.COLOR_BGR2GRAY)/255

    w, h = imm.shape[0], imm.shape[1]
    nw, nh = w//w_tar, h//h_tar
    
    l.append(np.sum(imm) / (w*h))

print(k)