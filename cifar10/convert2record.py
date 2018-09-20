import numpy as np
import tensorflow as tf
import cv2
from os import walk
from os.path import join
import time
import random


def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def main(argv):
    f = open("image_label_val.txt")
    data_list = f.readlines()

    writer = tf.python_io.TFRecordWriter("cifar10_224Val.tfrecord")

    count = 0
    for data_line in data_list:
        if len(data_line) > 5:
            line_str = str(data_line)
            line_objs = line_str.split(",")
            image_path = line_objs[0]
            label = int(line_objs[1])

            image_data = cv2.imread(image_path, cv2.IMREAD_COLOR)
            im_resize = cv2.resize(image_data, (224,224), interpolation=cv2.INTER_CUBIC)
            img_np = np.asarray(im_resize)

            img_raw = img_np.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'label' : _int64_feature(int(label)),
                'image_raw' : _bytes_feature(img_raw)
            }))
            writer.write(example.SerializeToString())
            count = count + 1

    writer.close()

if __name__ == '__main__':
    tf.app.run()




