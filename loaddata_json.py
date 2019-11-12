import numpy as np
import os
import json
from PIL import Image
import random
import torch
from skimage import io, transform, color
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

label_dic={'cell0':0,'cell1':1}

def loadjson(path, root_dir='./', transform=None):
    load_f = open(path, 'r')
    load_dict = json.load(load_f)
    img_name = load_dict['imgName']
    img_name = os.path.join(root_dir, img_name)
    image = io.imread(img_name)
    if transform:
            image=transform(image)
    d1,d2,d3=image.shape
    image=image.reshape([1,d1,d2,d3])
    objs = load_dict['objs']
    annotations  = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
    for i in range(len(objs)):
        annotations['labels'] = np.concatenate([annotations['labels'], [label_dic[objs[i]['label']]]], axis=0)
        annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                objs[i]['xmin'],
                objs[i]['ymin'],
                objs[i]['xmax'],
                objs[i]['ymax'],
            ]]], axis=0)
    sample = {'image': image, 'annotations': annotations}
    return sample

