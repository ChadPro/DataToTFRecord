# Copyright 2018 The LongYan. All Rights Reserved.
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle
import cv2
import sys
from os.path import join

IMG_SIZE = [3,32,32]

DATA_PATH = "../../Datasets/Cifar10/cifar-10-batches-py"
# each 10000 nums and image.shape = 3072 = (32,32,3)
TRAIN_LIST = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]   
TRAIN_LIST = [join(DATA_PATH, p) for p in TRAIN_LIST]
TEST_DATA_PATH = "../../Datasets/Cifar10/cifar-10-batches-py/test_batch"    # 10000 nums data
TRAIN_TFRECORD_PATH = "./Cifar10_32Train.tfrecord"
TEST_TFRECORD_PATH = "./Cifar10_32Test.tfrecord"

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def unpickle(file):
    with open(file, "rb") as fo:
        dic = pickle.load(fo, encoding="bytes")
    return dic

def convert(convert_type):

    if convert_type == "Test":
        writer = tf.python_io.TFRecordWriter(TEST_TFRECORD_PATH)
        data = unpickle(TEST_DATA_PATH)
        imgs = data[b"data"]
        labels = data[b"labels"]
        nums = len(labels)

        count = 0
        for i,img in enumerate(imgs):
            label = int(labels[i])
            img = np.reshape(img, IMG_SIZE)
            img = img.transpose([1,2,0])
            img_raw = img.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'label' : _int64_feature(int(label)),
                'image_raw' : _bytes_feature(img_raw)
            }))
            writer.write(example.SerializeToString())
            count = count + 1

            sys.stdout.write('\r>> Converting Test image %d/%d' % (count, nums))
            sys.stdout.flush()
        
        writer.close()

    if convert_type == "Train":
        writer = tf.python_io.TFRecordWriter(TRAIN_TFRECORD_PATH)
        count = 0
        for one_path in TRAIN_LIST:
            data = unpickle(one_path)
            imgs = data[b"data"]
            labels = data[b"labels"]

            for i,img in enumerate(imgs):
                label = int(labels[i])
                img = np.reshape(img, IMG_SIZE)
                img = img.transpose([1,2,0])
                img_raw = img.tostring()

                example = tf.train.Example(features=tf.train.Features(feature={
                    'label' : _int64_feature(int(label)),
                    'image_raw' : _bytes_feature(img_raw)
                }))
                writer.write(example.SerializeToString())
                count = count + 1

                sys.stdout.write('\r>> Converting Train image %d/%d' % (count, 50000))
                sys.stdout.flush()
        
        writer.close()



def main(argv):
    print("### Start Convert Train Dataset ###############")
    convert("Train")
    print(" ")
    print("### Start Convert Test Dataset ################")
    convert("Test")
    print(" ")

if __name__ == '__main__':
    tf.app.run()