import torch
import numpy as np
import skimage.io as io
from loaddata import COCO_Dataset, loaddata
from torchvision import datasets, models, transforms
from retinanet import Retinanet
from anchors import anchor_targets_bbox
from losses import Focalloss, Smooth_l1_loss
from filter_detections import bbox_transform_inv, clipBoxes, filter_detections

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

model=Retinanet(80)
model = model.to(device)
l1=Focalloss()
l2=Smooth_l1_loss()

model.fix_backbone()

optimizer = torch.optim.Adam(model.parameters(),lr=0.000005)

batch_c=[]
batch_r=[]
batch_prec=[]
batch_prer=[]
for a in data_loader:
    img=a['image']
    ann=a['annotations']
    if len(ann['labels'])==0:
        continue
    img = img.to(device)
    pre_r, pre_c, all_anchors=model(img)
    y_r, y_c=anchor_targets_bbox(all_anchors,ann,80)
    y_r=torch.from_numpy(y_r)
    y_c=torch.from_numpy(y_c)
    y_r=y_r.to(device)
    y_c=y_c.to(device)
    batch_c.append(y_c)
    batch_r.append(y_r)
    batch_prec.append(pre_c)
    batch_prer.append(pre_r)
    if len(batch_c)>=1:
        optimizer.zero_grad()
        b_c=torch.cat(batch_c,axis=0)
        b_r=torch.cat(batch_r,axis=0)
        b_prec=torch.cat(batch_prec,axis=0)
        b_prer=torch.cat(batch_prer,axis=0)
        loss1=l1(b_prec,b_c)
        loss2=l2(b_prer,b_r)
        loss=loss1+loss2
        print(loss1,loss2,loss)
        loss.backward()
        optimizer.step()
        batch_c=[]
        batch_r=[]
        batch_prec=[]
        batch_prer=[]

model.free_backbone()

batch_c=[]
batch_r=[]
batch_prec=[]
batch_prer=[]
for a in data_loader:
    img=a['image']
    ann=a['annotations']
    if len(ann['labels'])==0:
        continue
    img = img.to(device)
    pre_r, pre_c, all_anchors=model(img)
    y_r, y_c=anchor_targets_bbox(all_anchors,ann,80)
    y_r=torch.from_numpy(y_r)
    y_c=torch.from_numpy(y_c)
    y_r=y_r.to(device)
    y_c=y_c.to(device)
    batch_c.append(y_c)
    batch_r.append(y_r)
    batch_prec.append(pre_c)
    batch_prer.append(pre_r)
    if len(batch_c)>=1:
        optimizer.zero_grad()
        b_c=torch.cat(batch_c,axis=0)
        b_r=torch.cat(batch_r,axis=0)
        b_prec=torch.cat(batch_prec,axis=0)
        b_prer=torch.cat(batch_prer,axis=0)
        loss1=l1(b_prec,b_c)
        loss2=l2(b_prer,b_r)
        loss=loss1+loss2
        print(loss1,loss2,loss)
        loss.backward()
        optimizer.step()
        batch_c=[]
        batch_r=[]
        batch_prec=[]
        batch_prer=[]

all_anchors=torch.from_numpy(all_anchors).float()
all_anchors=all_anchors.to(device)

pred_boxes=bbox_transform_inv(all_anchors,pre_r)
clip_boxes=clipBoxes(pred_boxes,img)
p=filter_detections(clip_boxes,pre_c)

torch.save(model.cpu().state_dict(),'model/retinacoco.pth')

model2=Retinanet(2)
pretrained_dict = torch.load('model/retinacoco.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['Subnet_c.output.weight','Subnet_c.output.bias']}
model2.load_state_dict(pretrained_dict, strict=False)

