# Copyright 2018 The LongYan. All Rights Reserved.
"""A script for gen tfrecord."""

from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import cv2
from os import walk
from os.path import join
import time
import random
import sys

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def convert(isTrain=True):
    list_file = ""
    record_name = ""
    if isTrain:
        list_file = "random_train_list.txt"
        record_name = "flower17_448Train.tfrecord"
    else:
        list_file = "random_val_list.txt"
        record_name = "flower17_448Val.tfrecord"

    f = open(list_file)
    data_list = f.readlines()

    writer = tf.python_io.TFRecordWriter(record_name)

    count = 0
    list_len = len(data_list)-1
    for data_line in data_list:
        if len(data_line) > 5:
            line_str = str(data_line)
            line_objs = line_str.split(",")
            image_path = line_objs[0]
            label = int(line_objs[1])

            image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
            im_resize = cv2.resize(image_data, (448,448), interpolation=cv2.INTER_CUBIC)
            img_np = np.asarray(im_resize)

            img_raw = img_np.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'label' : _int64_feature(int(label)),
                'image_raw' : _bytes_feature(img_raw)
            }))
            writer.write(example.SerializeToString())
            count = count + 1

            if isTrain:
                sys.stdout.write('\r>> Converting Train image %d/%d' % (count, list_len))
                sys.stdout.flush()
            else:
                sys.stdout.write('\r>> Converting Val image %d/%d' % (count, list_len))
                sys.stdout.flush()

    writer.close()

def main(argv):
    convert(isTrain=True)
    print "######"
    convert(isTrain=False)
    print "######"

if __name__ == '__main__':
    tf.app.run()




