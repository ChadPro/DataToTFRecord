## 0. 介绍

**转换**运行脚本为**convert2record.py**,依赖文件**pascalvoc\_to\_tfrecords.py**,**dataset\_utils.py**,**pascalvoc\_common,py**  
**读取**测试**read_image.py**,依赖文件**pascalvoc\_2012.py**,**voc\_humanlight.py**

## 1. 人行红绿灯数据
#### 1.1 convert
python tf_convert_data.py \
    --dataset_name='humanlight' \
    --dataset_dir='' \
    --output_name='humanlight_train' \
    --output_dir=''
#### 1.2 read test
python read_image.py

## 2. VOC-21-class
#### 1.1 convert
python tf_convert_data.py \
    --dataset_name='pascalvoc' \
    --dataset_dir='' \
    --output_name='what_train' \
    --output_dir=''
#### 1.2 read test
