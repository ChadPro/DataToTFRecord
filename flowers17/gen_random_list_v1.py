# Copyright 2018 The Candela. All Rights Reserved.

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
parser.add_argument('--data_path', type=str, help='Image path.')
args = parser.parse_args()

def read_images(path):

    train_file_name = "random_train_list.txt"
    val_file_name = "random_val_list.txt"

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

    train_random_image = random_image[0:1000]
    train_random_label = random_label[0:1000]

    val_random_image = random_image[1000:]
    val_random_label = random_label[1000:]

    f = open(train_file_name, "w")
    for i in range(1000):
        txt_line = str(train_random_image[i]) + "," + str(train_random_label[i]) + "\n"
        f.write(txt_line)
    f.close()

    f = open(val_file_name, "w")
    for i in range(num_images-1000):
        txt_line = str(val_random_image[i]) + "," + str(val_random_label[i]) + "\n"
        f.write(txt_line)
    f.close()

def run():
    read_images(args.data_path)

if __name__ == '__main__':
    run()