# -*- coding: utf-8 -*-
import os
import glob
import random

root = os.getcwd()
data_annotated_path = os.path.join(root, 'data_annotated')
txt_save_path = root
json_file_path = glob.glob(os.path.join(data_annotated_path, '*.json'))
img_file_path = glob.glob(os.path.join(data_annotated_path, '*.bmp'))

assert len(json_file_path) == len(img_file_path)

trainval_percent = 1  # No test sample
train_percent = 0.9

num = len(json_file_path)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

f_trainval = open(os.path.join(root, 'trainval2007.txt'), 'w')
f_train = open(os.path.join(root, 'train2007.txt'), 'w')
f_val = open(os.path.join(root, 'val2007.txt'), 'w')
f_test = open(os.path.join(root, 'test2007.txt'), 'w')

for i in list:
    name = os.path.basename(json_file_path[i]).split('.')[0] + '\n'
    if i in trainval:
        f_trainval.write(name)
        if i in train:
            f_train.write(name)
        else:
            f_val.write(name)
    else:
        f_test.write(name)
f_trainval.close()
f_train.close()
f_val.close()
f_test.close()
print('Create_txt Done')
