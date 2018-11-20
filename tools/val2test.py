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

input_path = "/home/taylor/Mirror-Segmentation/data_640/val/"
output_path = "/home/taylor/SemirNet/data/test/"

imglist = os.listdir(input_path + "image")

for i, imgname in enumerate(imglist):
    print(i)
    original_num = imgname.split("_")[0]
    original_ends = imgname.split("_")[1]
    num = 1234 + int(original_num)
    new_name = str(num) + "_" + original_ends

    image = skimage.io.imread(input_path + "image/" + imgname)
    skimage.io.imsave(output_path + "image/" + new_name, image)

    mask = Image.open(input_path + "mask/" + imgname[:-4] + "_json/label8.png")
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

    skimage.io.imsave(output_path + "mask/" + new_name[:-4] + ".png", gt_mask)



