"""
  @Time    : 2018-11-20 00:00
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : SemirNet
  @File    : mask.py
  @Function: 
  
"""
import os
import numpy as np
from PIL import Image
import skimage.io

mask_path = "/home/taylor/Mirror-Segmentation/data_640/test/mask/"
output_path = "/home/taylor/SemirNet/data/test/mask/"

masklist = os.listdir(mask_path)

for i, maskname in enumerate(masklist):
    print(i)
    mask = Image.open(mask_path + maskname + "/label8.png")
    width, height = mask.size
    num_obj = np.max(mask)

    gt_mask = np.zeros([height, width], dtype=np.uint8)
    for index in range(num_obj):
        """j is row and i is column"""
        for i in range(width):
            for j in range(height):
                at_pixel = mask.getpixel((i, j))
                if at_pixel == index + 1:
                    gt_mask[j, i] = 255

    skimage.io.imsave(output_path + maskname[:-5] + ".png", gt_mask)



