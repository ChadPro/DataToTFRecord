# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import cv2
import cifar100_224

images, onehot_labels, labels = cifar10_224.inputs("./cifar100_224Train.tfrecord", "./cifar100_224Val.tfrecord", "Train", 10, None)

with tf.Session() as sess:
    imgs, clas = sess.run([images, labels])

    for i, img in enumerate(imgs):
        cv2.imwrite(str(i)+".jpg", img)
        print str(clas[i])