# -*- coding: utf-8 -*-
# Copyright 2018 The LongYan. All Rights Reserved.
from __future__ import absolute_import
from __future__ import division

import numpy as np
import tensorflow as tf
import cv2
import voc_humanlight
import pascalvoc_2012
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('datatype', type=str, help='')
parser.add_argument('datapath', type=str, help='')
args = parser.parse_args()

if args.datatype == 'voc21':
    image, shape, boxes, label = pascalvoc_2012.inputs(args.datapath, "", "Train", None)
elif args.datatype == 'humanlight':
    image, shape, boxes, label = voc_humanlight.inputs(args.datapath, "", "Train", None)

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