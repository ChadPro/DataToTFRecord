# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import cv2
import cifar10_32

images, onehot_labels, labels = cifar10_32.inputs("./Cifar10_32Train.tfrecord", "./Cifar10_32Test.tfrecord", "Train", 10, None)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    imgs, clas = sess.run([images, labels])

    for i, img in enumerate(imgs):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(i)+".jpg", img)
        print(str(clas[i]))

    coord.request_stop()
    coord.join(threads)