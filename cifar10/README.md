## Cifar10数据下载
网站

## 处理Cifar10数据为TFRecord
### 1. 生成数据随机列表
用脚本读取数据文件结构，生成对应的图片地址＋类别的列表，并将其随机排布(注意根据图片文件的存储形式应适当改写脚本):
```python
python gen_random_list.py --train_path="" --val_path=""
```
### 2. 生成TFRecord
读取文件列表txt文件并将其中的图片数据转成tfrecord格式：
```python
python convert2record.py
```

### 3. 读取测试
从生成的tfrecord数据中读取数据测试：
```python
python read_test.py
```