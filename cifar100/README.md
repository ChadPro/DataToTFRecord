## Cifar10数据下载
网站

## 处理Cifar10数据为TFRecord
### 1. 生成数据随机列表
```python
python gen_random_list.py --train_path="" --val_path=""
```
### 2. 生成TFRecord
```python
python convert2record.py
```

### 3. 读取测试
```python
python read_test.py
```