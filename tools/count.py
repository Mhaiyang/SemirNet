"""
  @Time    : 2018-11-19 23:30
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : SemirNet
  @File    : count.py
  @Function: 
  
"""
import os
import numpy as np
import skimage.io

data_path = "/home/taylor/Mirror-Segmentation/data_640/train/image/"

imglist = os.listdir(data_path)

count_high = 0
count_wide = 0
for i, imgname in enumerate(imglist):
    print(i)
    image = skimage.io.imread(data_path + imgname)
    h = np.shape(image)[0]
    w = np.shape(image)[1]
    if h > w:
        count_high += 1
    else:
        count_wide += 1
print(count_high)
print(count_wide)