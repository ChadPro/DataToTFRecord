# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

"""
python tf_convert_data.py \
    --dataset_name=pascalvoc \
    --dataset_dir=/tmp/pascalvoc \
    --output_name=pascalvoc \
    --output_dir=/tmp/
```
"""
import tensorflow as tf
import pascalvoc_to_tfrecords

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_name', 'pascalvoc',
    'The name of the dataset to convert.')
tf.app.flags.DEFINE_string(
    'data_dir', None,
    'Directory where the original dataset is stored.')
tf.app.flags.DEFINE_string(
    'output_name', 'pascalvoc',
    'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string(
    'output_dir', './',
    'Output directory where to store TFRecords files.')

def main(_):
    if not FLAGS.data_dir:
        raise ValueError('You must supply the dataset directory with --data_dir')
    print('Data directory:', FLAGS.data_dir)
    print('Output directory:', FLAGS.output_dir)

    # if FLAGS.data_name == 'pascalvoc':
    #     pascalvoc_to_tfrecords.run(FLAGS.data_dir, FLAGS.output_dir, FLAGS.output_name)
    # else:
    #     raise ValueError('Data [%s] was not recognized.' % FLAGS.data_name)

    pascalvoc_to_tfrecords.run(FLAGS.data_dir, FLAGS.output_dir, FLAGS.output_name, dataname=FLAGS.data_name)

if __name__ == '__main__':
    tf.app.run()

