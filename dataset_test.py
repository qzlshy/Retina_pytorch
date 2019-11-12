import numpy as np
from PIL import Image
import random

from anchors import anchor_targets_bbox,anchors_for_shape,guess_shapes

dataDir='/state/partition1/long/coco2017'
train_dir='train2017'
train_annFile='{}/annotations/instances_{}.json'.format(dataDir,train_dir)

coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['id'] for cat in cats]
classes={}
for i in range(len(nms)):
	classes[nms[i]]=i


def get_train_ids():
	imgIds=coco.getImgIds()
	return imgIds

def g_data(imgId):
	
	img_name = coco.loadImgs(imgId)[0]['file_name']
	img = Image.open('%s/%s/%s'%(dataDir,dataType,img_name))
	img_group.append(img_group)
	annIds = coco.getAnnIds(imgIds=imgId)
	anns=coco.loadAnns(annIds)
	annotations = {'labels': np.empty((0,)), 'bboxes': np.empty((0, 4))}
	if len(annotations_ids) == 0
		return annotations

	for idx, a in enumerate(anns):
		if a['bbox'][2] < 1 or a['bbox'][3] < 1:
			continue
		annotations['labels'] = np.concatenate([annotations['labels'], [classes[(a['category_id'])]], axis=0)
		annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
			a['bbox'][0],
			a['bbox'][1],
			a['bbox'][0] + a['bbox'][2],
			a['bbox'][1] + a['bbox'][3],
		]]], axis=0)


		
