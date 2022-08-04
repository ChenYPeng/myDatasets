from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib
import cv2

matplotlib.use('Qt5Agg')
file = "hanfengseg/annotations/instances_train2017.json"
pic_path = "hanfengseg/JPEGImages/train2017/0018.jpg"
im = cv2.imread(pic_path)
plt.imshow(im)
plt.axis('off')
cc = COCO(file)
annIds = cc.getAnnIds(imgIds=18)
anns = cc.loadAnns(annIds)
cc.showAnns(anns)
plt.show()
