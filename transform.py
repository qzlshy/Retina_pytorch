import torch
import random
from PIL import Image
from torchvision import transforms


class Rescale(object):
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size
	def __call__(self, image, boxes):
		w, h = image.size
		if isinstance(self.output_size, int):
			if h > w:
				new_h, new_w = self.output_size * h / w, self.output_size
			else:
				new_h, new_w = self.output_size, self.output_size * w / h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)
		resize = transforms.Resize((new_h, new_w))
		img=resize(image)
		boxes = boxes * [new_w / w, new_h / h, new_w / w, new_h / h]
		return img, boxes


def random_Hflip(img, boxes):
	if random.random() < 0.5:
		w, h = img.size
		flip=transforms.RandomHorizontalFlip(p=1)
		img=flip(img)
		x_max = w - boxes[:, 0]
		x_min = w - boxes[:, 2]
		boxes[:,0]=x_min
		boxes[:,2]=x_max
	return img,boxes

def random_Vflip(img, boxes):
	if random.random() < 0.5:
		w, h = img.size
		flip=transforms.RandomVerticalFlip(p=1)
		img=flip(img)
		y_max = h - boxes[:, 1]
		y_min = h - boxes[:, 3]
		boxes[:,1]=y_min
		boxes[:,3]=y_max
	return img,boxes


