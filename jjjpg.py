import scipy.io as sio
import skimage.io
import hdf5storage
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from spectral import *
import spectral.io.envi as envi
import scipy.io as sio
import os
import pandas as pd
import h5py

plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
plt.rcParams['font.size'] = 12  # 字体大小
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
filename ='hinet'
savename='orinal'

#path1 = r"F:/2-Hami data SR/dfjproject/test_develop_code/exp/"+filename +"/health31.mat"
#path2 = r"F:/2-Hami data SR/dfjproject/test_develop_code/exp/"+filename +"/symptomatic124.mat"
#path3 = r"F:/2-Hami data SR/dfjproject/test_develop_code/exp/"+filename +"/mildew74.mat"
path1 = r"F:/hami-13/test/health31.mat"
path2 = r"F:/hami-13/test/symptomatic124.mat"
path3 = r"F:/hami-13/test/mildew74.mat"
with h5py.File(path1, 'r') as mat:
    hyper = np.float32(np.array(mat['newmat']))
# print(hyper.shape)
hyper = hyper.swapaxes(1, 2)
filenum='health31'
for i in range(13):
    hyper1 = hyper[i][:][:]
    plt.imsave("F:/"+savename+ "/" + filenum + "-"+ str(i)+".jpg", hyper1)

with h5py.File(path2, 'r') as mat:
    hyper = np.float32(np.array(mat['newmat']))
# print(hyper.shape)
hyper = hyper.swapaxes(1, 2)
filenum='symptomatic124'
for i in range(13):
    hyper1 = hyper[i][:][:]
    plt.imsave("F:/"+savename+ "/" + filenum + "-"+ str(i)+".jpg", hyper1)
with h5py.File(path3, 'r') as mat:
    hyper = np.float32(np.array(mat['newmat']))
# print(hyper.shape)
hyper = hyper.swapaxes(1, 2)
filenum='mildew74'
for i in range(13):
    hyper1 = hyper[i][:][:]
    plt.imsave("F:/"+savename+ "/" + filenum + "-"+ str(i)+".jpg", hyper1)

