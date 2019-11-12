import torch
import numpy as np
import skimage.io as io
from loaddata import COCO_Dataset, loaddata
from torchvision import datasets, models, transforms
from retinanet import Retinanet


data_transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

dataDir='/state/partition1/long/coco2017/images/train2017'
annFile='/state/partition1/long/coco2017/annotations/instances_train2017.json'
testDir='/state/partition1/long/coco2017/images/test2017'
test_annFile='/state/partition1/long/coco2017/annotations/instances_val2017.json'

datasets_train=COCO_Dataset(annFile, dataDir ,transform=data_transforms)
datasets_test=COCO_Dataset(test_annFile, testDir ,transform=data_transforms)
data_loader=loaddata(datasets_train)

for a in data_loader:
    t=a
    break


model=Retinanet(80)

a=datasets_train[0]
x=a['image']
d1,d2,d3=x.shape
x=x.reshape([1,d1,d2,d3])
r=model(x)
