# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import time
from os import walk
from os.path import join
import sys
import cv2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import array_ops

TRAIN_FILE = "humanlight_train_%03d.tfrecord"

def decode_image(image_buffer):
    return image_ops.decode_jpeg(image_buffer, 3)

def decode_boxes(ymin, xmin, ymax, xmax):
    ymin_v = ymin.values
    xmin_v = xmin.values
    ymax_v = ymax.values
    xmax_v = xmax.values

    sides = []
    sides.append(array_ops.expand_dims(ymin_v, 0))
    sides.append(array_ops.expand_dims(xmin_v, 0))
    sides.append(array_ops.expand_dims(ymax_v, 0))
    sides.append(array_ops.expand_dims(xmax_v, 0))

    bounding_box = array_ops.concat(sides, 0)
    return array_ops.transpose(bounding_box)

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
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64)
    })

    image = decode_image(keys_to_features['image/encoded'])
    shape = keys_to_features['image/shape']
    boxes = decode_boxes(keys_to_features['image/object/bbox/ymin'],
                        keys_to_features['image/object/bbox/xmin'],
                        keys_to_features['image/object/bbox/ymax'],
                        keys_to_features['image/object/bbox/xmax'])
    label = keys_to_features['image/object/bbox/label']

    return image, shape, boxes, label

def inputs(train_path, val_path, data_set, num_epochs):
    data_file_num = 0
    if not num_epochs:
        num_epochs = None
    if data_set == 'Train':
        read_file = train_path
        for (root, dirs, files) in walk(read_file):
            data_file_num = len(files)
    else:
        read_file = val_path

    with tf.name_scope('tfrecord_input') as scope:

        file_lists = [join(train_path, TRAIN_FILE) % i for i in range(0, data_file_num)]
        filename_queue = tf.train.string_input_producer(file_lists, num_epochs=num_epochs)
        image, shape, boxes, label = read_and_decode(filename_queue)
    
    return image, shape, boxes, label