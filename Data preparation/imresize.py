#!/usr/bin/env python
# coding: utf-8


import cv2
import PIL
from PIL import Image
import os



i=0
for image_file_name in sorted(os.listdir('./imgs/')):
    i=i+1
    if image_file_name.endswith(".jpg"):
        im = Image.open('./imgs/'+image_file_name)
        new_width  = 224
        new_height = 224
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save(str(i) + '.jpg')

