import torch
import numpy as np
import skimage.io as io
from torchvision import datasets, models, transforms
from retinanet import Retinanet
from anchors import anchor_targets_bbox
from losses import Focalloss, Smooth_l1_loss
from filter_detections import bbox_transform_inv, clipBoxes, filter_detections
import os
import loaddata_json

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_transforms=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


root_dir='/state/partition1/long/2-2_B_img'
json_dir='/state/partition1/long/2-2_B_mask'
json_files=os.listdir(json_dir)
json_path=[]
for a in json_files:
    json_path.append(os.path.join(json_dir,a))

json_path_train=json_path[:10]
json_path_test=json_path[10:]

model=Retinanet(2)
pretrained_dict = torch.load('model/retinacoco.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['Subnet_c.output.weight','Subnet_c.output.bias']}
model.load_state_dict(pretrained_dict, strict=False)

model = model.to(device)
l1=Focalloss()
l2=Smooth_l1_loss()

optimizer = torch.optim.Adam(model.parameters(),lr=0.000005)

epoch1=100
epoch2=200

model.fix_backbone()
for e in range(epoch1):
    for j_file in json_path_train:
        optimizer.zero_grad()
        a=loaddata_json.loadjson(j_file,root_dir,data_transforms)
        img=a['image']
        ann=a['annotations']
        img = img.to(device)
        pre_r, pre_c, all_anchors=model(img)
        y_r, y_c=anchor_targets_bbox(all_anchors,ann,2)
        y_r=torch.from_numpy(y_r)
        y_c=torch.from_numpy(y_c)
        y_r=y_r.to(device)
        y_c=y_c.to(device)
        loss1=l1(pre_c,y_c)
        loss2=l2(pre_r,y_r)
        loss=loss1+loss2
        print(loss1,loss2,loss)
        loss.backward()
        optimizer.step()

model.free_backbone()
for e in range(epoch2):
    for j_file in json_path_train:
        optimizer.zero_grad()
        a=loaddata_json.loadjson(j_file,root_dir,data_transforms)
        img=a['image']
        ann=a['annotations']
        img = img.to(device)
        pre_r, pre_c, all_anchors=model(img)
        y_r, y_c=anchor_targets_bbox(all_anchors,ann,2)
        y_r=torch.from_numpy(y_r)
        y_c=torch.from_numpy(y_c)
        y_r=y_r.to(device)
        y_c=y_c.to(device)
        loss1=l1(pre_c,y_c)
        loss2=l2(pre_r,y_r)
        loss=loss1+loss2
        print(loss1,loss2,loss)
        loss.backward()
        optimizer.step()


import json
dic_rev={0:'cell0',1:'cell1'}

model.eval()

for j_file in json_path_test:
    a=loaddata_json.loadjson(j_file,root_dir,data_transforms)
    img=a['image']
    ann=a['annotations']
    img = img.to(device)
    pre_r, pre_c, all_anchors=model(img)
    all_anchors=torch.from_numpy(all_anchors).float()
    all_anchors=all_anchors.to(device)
    pred_boxes=bbox_transform_inv(all_anchors,pre_r)
    clip_boxes=clipBoxes(pred_boxes,img)
    p=filter_detections(clip_boxes,pre_c,score_threshold = 0.2,class_specific_filter = False, nms_threshold = 0.1,max_detections = 30)
    box_tmp=[]
    box_pre=p[0].data.cpu().numpy()
    label_pre=p[2].data.cpu().numpy()
    for i in range(len(label_pre)):
        t={"label":dic_rev[label_pre[i]],"xmin":int(box_pre[i][0]),"xmax":int(box_pre[i][2]),"ymin":int(box_pre[i][1]),"ymax":int(box_pre[i][3])}
        box_tmp.append(t)
    out_file_name=j_file.split('/')[-1].split('.')[0]+'v2e3_pre.json'
    objs={"imgName":j_file,"objs":box_tmp}
    with open(out_file_name,"w") as f_obj:
        json.dump(objs,f_obj)


