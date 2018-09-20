import numpy as np
import tensorflow as tf
import cv2
from os import walk
from os.path import join
import time
import random

def read_images(path):
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

    f = open("image_label_val.txt", "w")
    for i in range(num_images):
        txt_line = str(random_image[i]) + "," + str(random_label[i]) + "\n"
        f.write(txt_line)
    f.close()

read_images("validation/")