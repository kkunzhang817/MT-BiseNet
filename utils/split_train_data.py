# *coding:utf-8 *

import os
import random

trainval_percent = 0.2  # 可自行进行调节(设置训练和测试的比例是8：2)
train_percent = 1
xmlfilepath = r'D:\zkk_project\segmentation\segment\data\Leaf\train\aug\SegmentationClass_AUG'
txtsavepath = r'D:\zkk_project\segmentation\segment\data\Leaf\train\aug\JPEG_AUG'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

# ftrainval = open('ImageSets/Main/trainval.txt', 'w')
fval = open(r'D:\zkk_project\segmentation\segment\data\Leaf\train\aug\val.txt', 'w')
ftrain = open(r'D:\zkk_project\segmentation\segment\data\Leaf\train\aug\train.txt', 'w')
# fval = open('ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i] + '\n'
    if i in trainval:
        # ftrainval.write(name)
        if i in train:
            fval.write(name)
        # else:
        # fval.write(name)
    else:
        ftrain.write(name)

# ftrainval.close()
ftrain.close()
# fval.close()
fval.close()
