# Copyright 2018 The LongYan. All Rights Reserved.

"""A script for gen random list."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import cv2
from os import walk
from os.path import join
import time
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', type=str, help='Train image path.')
parser.add_argument('--val_path', type=str, help='Validation image path.')
args = parser.parse_args()

def read_images(path, isTrain=True):
    txt_name = "error.txt"

    if isTrain:
        txt_name = "random_train_list.txt"
    else:
        txt_name = "random_val_list.txt"

    image_all_path = []
    label_all = []

    image_file_list = []
    for (root, dirs, files) in walk(path):
        if root == path:
            image_file_list = dirs
    
    # 10 clases image
    for image_class_file in image_file_list:
        real_image_class_file_path = join(path, image_class_file)
        
        # one class all images
        for (root, dirs, files) in walk(real_image_class_file_path):
            jpg_list = files
            for i,jpg_path in enumerate(jpg_list):
                jpg_list[i] = join(real_image_class_file_path, jpg_path)
                label_all.append(int(image_class_file))
            image_all_path.extend(jpg_list)
        
    num_images = len(image_all_path)
    
    np_image = np.array(image_all_path)
    np_label = np.array(label_all)

    random_index = range(num_images)
    random.shuffle(random_index)
    random_image = np_image[random_index]
    random_label = np_label[random_index]

    f = open(txt_name, "w")
    for i in range(num_images):
        txt_line = str(random_image[i]) + "," + str(random_label[i]) + "\n"
        f.write(txt_line)
    f.close()

def run():
    read_images(args.train_path, isTrain=True)
    read_images(args.val_path, isTrain=False)

if __name__ == '__main__':
    run()