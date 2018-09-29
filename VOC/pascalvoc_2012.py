# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import time
from scipy.misc import imread,imresize
from os import walk
from os.path import join
import sys
import cv2

# Image 
IMG_SIZE = 448
IMG_CHANNELS = 3

TRAIN_FILE = "voc_2012_train_%03d.tfrecord"

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)

    keys_to_features = tf.parse_single_example(serialized_example,features={
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    })

    # image = tf.decode_raw(keys_to_features['image_raw'],tf.uint8)
    # label = tf.cast(keys_to_features['label'],tf.int32)
    # image.set_shape([IMG_WIDTH*IMG_HEIGHT*IMG_CHANNELS])
    # image = tf.reshape(image,[IMG_SIZE,IMG_SIZE,IMG_CHANNELS])
    return keys_to_features['image/object/bbox/label'].values

def inputs(train_path, val_path, data_set,batch_size,num_epochs):
    data_file_num = 0
    if not num_epochs:
        num_epochs = None
    if data_set == 'train':
        read_file = train_path
        for (root, dirs, files) in walk(read_file):
            data_file_num = len(files)
    else:
        read_file = val_path

    with tf.name_scope('tfrecord_input') as scope:
        file_lists = [TRAIN_FILE % i for i in range(0,data_file_num)]
        for j in range(data_file_num):
            file_lists[j] = join(train_path, file_lists[j])

        filename_queue = tf.train.string_input_producer(file_lists, num_epochs=num_epochs)
        label = read_and_decode(filename_queue)
        labels = tf.train.shuffle_batch([label], batch_size=batch_size, num_threads=64, capacity=5000, min_after_dequeue=3000)

    return labels