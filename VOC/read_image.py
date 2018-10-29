# Copyright 2018 The Candela. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import cv2
import voc_humanlight

FLAGS = tf.app.flags.FLAGS

image, shape, boxes, label = voc_humanlight.inputs("./tfrecords", "", "Train", None)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    rimage, rshape, rboxes, rlabel = sess.run([image, shape, boxes, label])
    rimage = rimage[...,::-1]
    rimage = np.array(rimage)
    for i,bound_box in enumerate(rboxes):
        cv2.rectangle(rimage, (int(bound_box[1]*rshape[1]),int(bound_box[0]*rshape[0])),(int(bound_box[3]*rshape[1]),int(bound_box[2]*rshape[0])),(255,0,0),3)
        cv2.putText(rimage, str(rlabel.values[i]), (int(bound_box[1]*rshape[1]),int(bound_box[0]*rshape[0])-10), cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255), 3)
    cv2.imwrite("hehe.jpg", rimage)

    coord.request_stop()
    coord.join(threads)