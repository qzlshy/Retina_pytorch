import numpy as np
import os
from PIL import Image
import random
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils



class COCO_Dataset(Dataset):
    def __init__(self, json_file, root_dir, transform=None):
        self.root_dir=root_dir
        self.coco=COCO(json_file)
        self.transform = transform
        self.imge_ids=self.coco.getImgIds()
        self.get_class()

    def get_class(self):
        cats = self.coco.loadCats(self.coco.getCatIds())
        cats.sort(key=lambda x: x['id'])
        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in cats:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.imge_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img=self.coco.loadImgs(self.imge_ids[idx])[0]['file_name']
        img_name = os.path.join(self.root_dir, img)
        image = Image.open(img_name)
        if image.mode == 'L':
            image = image.convert("RGB")
        if self.transform:
            image=self.transform(image)
        annotations_ids = self.coco.getAnnIds(imgIds=self.imge_ids[idx], iscrowd=False)
        annotations  = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
        if len(annotations_ids) == 0:
            sample = {'image': image, 'annotations': annotations}
            return sample
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            annotations['labels'] = np.concatenate([annotations['labels'], [self.coco_labels_inverse[a['category_id']]]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][0] + a['bbox'][2],
                a['bbox'][1] + a['bbox'][3],
            ]]], axis=0)
        sample = {'image': image, 'annotations': annotations}
        return sample

class loaddata:
    def __init__(self,dataset,shuffle=True):
        self.dataset=dataset
        self.ids=list(range(len(dataset)))
        self.p=0
        self.shuffle=shuffle
        if shuffle:
            random.shuffle(self.ids)

    def __iter__(self):
        return self

    def __next__(self):
        self.p+=1
        if self.p>len(self.ids):
            self.p=0
            if self.shuffle:
                random.shuffle(self.ids)
            raise StopIteration()
        data = self.dataset[self.ids[self.p-1]]
        d1,d2,d3=data['image'].shape
        data['image']=data['image'].reshape([1,d1,d2,d3])
        return data
