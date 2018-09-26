# Copyright 2018 The Candela. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import cv2
import flowers17_448

images, onehot_labels, labels = flowers17_448.inputs("./flower17_448Train.tfrecord", "./flower17_448Val.tfrecord", "Train", 10, None)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    imgs, clas = sess.run([images, labels])

    for i, img in enumerate(imgs):
        cv2.imwrite(str(i)+".jpg", img)
        print str(clas[i])

    coord.request_stop()
    coord.join(threads)