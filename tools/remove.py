"""
  @Time    : 2018-11-20 04:28
  @Author  : TaylorMei
  @Email   : mhy845879017@gmail.com
  
  @Project : SemirNet
  @File    : remove.py
  @Function: 
  
"""
import os

image_folder = "/home/taylor/SemirNet/data/test/image/"
mask_folder = "/home/taylor/SemirNet/data/test/mask/"

masklist = os.listdir(mask_folder)

for i, maskname in enumerate(masklist):
    image_path = image_folder + maskname[:-4] + ".jpg"
    if not os.path.exists(image_path):
        os.remove(mask_folder + maskname)
        print("remove {}".format(mask_folder+maskname))
