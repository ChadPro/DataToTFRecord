# -- coding: utf-8 --
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import tensorflow as tf

import tf_utils
import time
import numpy as np
import traffic_light_dataset_15
import ssd_mobilenet_300_v1
import traffic_light_preprocessing
slim = tf.contrib.slim

batch_size = 5

def train():
    dataset = traffic_light_dataset_15.get_dataset("", batch_size, 21)
    provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=1,
                    common_queue_capacity=20 * batch_size,
                    common_queue_min=10 * batch_size,
                    shuffle=True)
    [image, shape, glabels, gbboxes] = provider.get(['image', 'shape', 'object/label', 'object/bbox'])

    ssd_net = ssd_mobilenet_300_v1.SSDNet()
    ssd_anchors = ssd_net.anchors((224,224))

    image, glabels, gbboxes = traffic_light_preprocessing.preprocess_for_train(image, glabels, gbboxes, color_space="rgb", out_shape=(224,224))
    gclasses, glocalisations, gscores = ssd_net.bboxes_encode(glabels, gbboxes, ssd_anchors)
    batch_shape = [1] + [len(ssd_anchors)] * 3
    # Training batches and queue.
    r = tf.train.batch(tf_utils.reshape_list([image, gclasses, glocalisations, gscores]),
                batch_size=batch_size,
                num_threads=2,
                capacity=5 * batch_size)
    b_image, b_gclasses, b_glocalisations, b_gscores = tf_utils.reshape_list(r, batch_shape)
    # Intermediate queueing: unique batch computation pipeline for all
    # GPUs running the training.
    global_step = tf.Variable(0, trainable=False)
    batch_queue = slim.prefetch_queue.prefetch_queue(tf_utils.reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]), capacity=2 * 1)
    b_image, b_gclasses, b_glocalisations, b_gscores = tf_utils.reshape_list(batch_queue.dequeue(), batch_shape)
    
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)  
        init_op = tf.global_variables_initializer()
        sess.run(init_op)   
        


        coord.request_stop()
        coord.join(threads)

def main(argv=None):
    train()

if __name__== '__main__': 
    tf.app.run()