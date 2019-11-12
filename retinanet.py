import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet101
from anchors import anchors_for_shape

class FPN(nn.Module):
	def __init__(self,backbone=resnet101(pretrained=True)):
		super(FPN, self).__init__()
		self.backbone=backbone
		self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
		self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
		self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
		self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
		self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
		self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
		self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

	def _upsample_add(self, x, y):
		_,_,H,W = y.size()
		return F.interpolate(x, size=(H,W), mode='bilinear') + y

	def forward(self, x):
		c3,c4,c5=self.backbone.get_feature(x)
		p6 = self.conv6(c5)
		p7 = self.conv7(F.relu(p6))
		p5 = self.latlayer1(c5)
		p4 = self._upsample_add(p5, self.latlayer2(c4))
		p4 = self.toplayer1(p4)
		p3 = self._upsample_add(p4, self.latlayer3(c3))
		p3 = self.toplayer2(p3)
		return [p3, p4, p5, p6, p7]


class Subnet_c(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Subnet_c, self).__init__()
        self.num_classes=num_classes
        self.conv1=nn.Conv2d(num_features,256,kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(256, 9*num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self,x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = self.output_act(out)
        out = out.permute(0, 2, 3, 1)
        out = torch.reshape(out,(-1,self.num_classes))
        return out

class Subnet_r(nn.Module):
    def __init__(self, num_features):
        super(Subnet_r, self).__init__()
        self.conv1 = nn.Conv2d(num_features, 256, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(256, 9*4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        out = torch.reshape(out,(-1,4))
        return out


class Retinanet(nn.Module):
    def __init__(self,num_classes):
        super(Retinanet, self).__init__()
        self.num_classes=num_classes
        self.fpn=FPN()
        self.Subnet_c=Subnet_c(256,num_classes)
        self.Subnet_r=Subnet_r(256)

    def fix_backbone(self):
        for p in self.fpn.backbone.parameters():
            p.requires_grad = False

    def free_backbone(self):
        for p in self.fpn.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        features=self.fpn(x)
        shapes=[[feature.shape[2],feature.shape[3]] for feature in features]
        regression = torch.cat([self.Subnet_r(feature) for feature in features],dim=0)
        classification = torch.cat([self.Subnet_c(feature) for feature in features], dim=0)
        all_anchors=anchors_for_shape(shapes)
        return regression, classification, all_anchors
        


