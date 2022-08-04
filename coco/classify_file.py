import shutil
import os
import os.path as osp

sets = ['train2017', 'val2017', 'test2017']
for image_set in sets:
    if osp.exists(image_set):
        shutil.rmtree(image_set)
        print('Deleted previous %s file and created a new one' % image_set)
    os.makedirs(image_set)
    json_path = '%s' % image_set
    if osp.exists(json_path):
        shutil.rmtree(json_path)
        print('Deleted previous %s file and created a new one' % json_path)
    os.makedirs(json_path)
    image_ids = open('./%s.txt' % image_set).read().strip().split()

    for image_id in image_ids:
        img = 'data_annotated/%s.bmp' % image_id
        json = 'data_annotated/%s.json' % image_id
        shutil.copy(img, image_set)
        shutil.copy(json, '%s/' % image_set)
print("Done")
