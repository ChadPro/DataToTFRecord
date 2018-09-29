import os

import tensorflow as tf
from datasets import dataset_utils

slim = tf.contrib.slim

TRAFFIC_LIGHT_LABELS = {
    'none': 0,
    'green': 1,
    'green_ahead': 2,
    'green_left': 3,
    'green_right': 4,
    'yellow': 5,
    'yellow_ahead': 6,
    'yellow_left': 7,
    'yellow_right': 8,
    'red': 9,
    'red_ahead': 10,
    'red_left': 11,
    'red_right': 12,
    'human_green': 13,
    'human_red': 14
}

def get_dataset(dataDir,dataSize,numClasses):
    reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature([], tf.string, default_value=''),
        'image/format': tf.FixedLenFeature([], tf.string, default_value='jpeg'),
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
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
            data_sources=dataDir,
            reader=reader,
            decoder=decoder,
            num_samples=230,
            items_to_descriptions={},
            num_classes=numClasses)