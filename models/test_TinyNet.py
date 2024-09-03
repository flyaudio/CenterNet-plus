import os
import cv2
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import setup
if __name__ == '__main__':
	setup.setCurPath(__file__)
import util


class HeatmapLoss(nn.Module):
	def __init__(self, alpha=2, beta=4, reduction='mean'):
		super().__init__()
		self.alpha = alpha
		self.beta = beta
		self.reduction = reduction

	def forward(self, logits:torch.tensor, targets:torch.tensor) -> torch.tensor:
		"""
		:param logits:[b,512/16,80]
		:param targets: [b,512/16,80]
		:return:loss(sum | mean)
		"""
		inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)
		pos_ind = (targets == 1.0).float()
		neg_ind = (targets != 1.0).float()
		pos_loss = -pos_ind * (1.0 - inputs)**self.alpha * torch.log(inputs)
		neg_loss = -neg_ind * (1.0 - targets)**self.beta * (inputs)**self.alpha * torch.log(1.0 - inputs)
		loss = pos_loss + neg_loss

		if self.reduction == 'mean':
			batch_size = loss.size(0)
			loss = torch.sum(loss) / batch_size

		if self.reduction == 'sum':
			loss = torch.sum(loss) / batch_size

		return loss


class ConvResidual(nn.Module):
	def __init__(self, inchannel, outchannel, k,s=1,p=1,d=1):
		super().__init__()
		self.convs = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, (k, k), stride=s, padding=p, dilation=d, groups=1, bias=True),
			nn.LeakyReLU(0.1, inplace=False)
		)

	def forward(self,x):
		return self.convs(x)


class ConvResidualWithNorm(nn.Module):
	def __init__(self, inchannel, outchannel, k=3, s=1, p=0, d=1):
		super().__init__()
		self.convs = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, (k, k), stride=s, padding=p, dilation=d, groups=1, bias=True),
			nn.LeakyReLU(0.1, inplace=False),
			nn.BatchNorm2d(outchannel)
		)
		self.shortcut = nn.Sequential()

	def forward(self,x):
		return self.shortcut(x) + self.convs(x)

class ConvResidualWithoutNorm(nn.Module):
	def __init__(self, inchannel, outchannel, k=3, s=1, p=0, d=1):
		super().__init__()
		self.convs = nn.Sequential(
			nn.Conv2d(inchannel, outchannel, (k, k), stride=s, padding=p, dilation=d, groups=1, bias=True),
			nn.LeakyReLU(0.1, inplace=False),
		)
		self.shortcut = nn.Sequential()

	def forward(self,x):
		return self.shortcut(x) + self.convs(x)

class Backbone(nn.Module):
	def __init__(self):
		super().__init__()
		"""
name=backbone.features.0_0.conv1.weight, shape=torch.Size([16, 3, 7, 7])
name=backbone.features.0_0.conv1.bias, shape=torch.Size([16])
name=backbone.features.0_1.conv1.weight, shape=torch.Size([16, 16, 1, 1])
name=backbone.features.0_1.conv1.bias, shape=torch.Size([16])
name=backbone.features.0_1.bn.weight, shape=torch.Size([16])
name=backbone.features.0_1.bn.bias, shape=torch.Size([16])
name=backbone.features.1_0.conv1.weight, shape=torch.Size([32, 16, 4, 4])
name=backbone.features.1_0.conv1.bias, shape=torch.Size([32])
name=backbone.features.1_1.conv1.weight, shape=torch.Size([32, 32, 1, 1])
name=backbone.features.1_1.conv1.bias, shape=torch.Size([32])
name=backbone.features.1_1.bn.weight, shape=torch.Size([32])
name=backbone.features.1_1.bn.bias, shape=torch.Size([32])
name=backbone.features.2_0.conv1.weight, shape=torch.Size([64, 32, 4, 4])
name=backbone.features.2_0.conv1.bias, shape=torch.Size([64])
name=backbone.features.2_1.conv1.weight, shape=torch.Size([64, 64, 1, 1])
name=backbone.features.2_1.conv1.bias, shape=torch.Size([64])
name=backbone.features.2_1.bn.weight, shape=torch.Size([64])
name=backbone.features.2_1.bn.bias, shape=torch.Size([64])
name=backbone.features.3_0.conv1.weight, shape=torch.Size([128, 64, 4, 4])
name=backbone.features.3_0.conv1.bias, shape=torch.Size([128])
name=backbone.features.3_1.conv1.weight, shape=torch.Size([128, 128, 1, 1])
name=backbone.features.3_1.conv1.bias, shape=torch.Size([128])
name=backbone.features.3_1.bn.weight, shape=torch.Size([128])
name=backbone.features.3_1.bn.bias, shape=torch.Size([128])
name=backbone.features.4_0.conv1.weight, shape=torch.Size([256, 128, 4, 4])
name=backbone.features.4_0.conv1.bias, shape=torch.Size([256])
name=backbone.features.4_1.conv1.weight, shape=torch.Size([256, 256, 1, 1])
name=backbone.features.4_1.conv1.bias, shape=torch.Size([256])
name=backbone.features.4_2.conv1.weight, shape=torch.Size([256, 256, 1, 1])
name=backbone.features.4_2.conv1.bias, shape=torch.Size([256])
		"""
		# self.feature_00 = Conv7x7()
		self.feature_00 = ConvResidual(3,16,k=7,s=2,p=3,d=1) #[1,16,256,320]
		self.feature_01 = ConvResidualWithNorm(16, 16, k=1, s=1, p=0, d=1) #[1,16,256,320]

		i=16
		o=32
		self.feature_10 = ConvResidual(i, o, k=4, s=2, p=1, d=1) #[1,32,128,160]
		self.feature_11 = ConvResidualWithNorm(o,o,k=1,s=1,p=0,d=1) #[1,32,128,160]

		i=32
		o=64
		self.feature_20 = ConvResidual(i, o, k=4, s=2, p=1, d=1) #[1,64,64,80]
		self.feature_21 = ConvResidualWithNorm(o,o,k=1,s=1,p=0,d=1) #[1,64,64,80]

		i=64
		o=128
		self.feature_30 = ConvResidual(i, o, k=4, s=2, p=1, d=1) #[1,128,32,40]
		self.feature_31 = ConvResidualWithNorm(o,o,k=1,s=1,p=0,d=1) #[1,128,32,40]

		i=128
		o=256
		self.feature_40 = ConvResidual(i, o, k=4, s=2, p=1, d=1) #[1,256,16,20]
		self.feature_41 = ConvResidualWithoutNorm(o,o,k=1,s=1,p=0,d=1) #[1,256,16,20]
		self.feature_42 = ConvResidualWithoutNorm(o,o,k=1,s=1,p=0,d=1) #[1,256,16,20]

	def forward(self,
				x: torch.tensor) -> (torch.tensor, torch.tensor):
		x = self.feature_01(self.feature_00(x))
		x = self.feature_11(self.feature_10(x))
		x = self.feature_21(self.feature_20(x))
		x3 = self.feature_31(self.feature_30(x))
		x4 = self.feature_41(self.feature_40(x3))
		return self.feature_42(x4), x3


class Neck(nn.Module):
	"""
name=neck.upsample.1.weight, shape=torch.Size([128, 256, 3, 3])
name=neck.upsample.1.bias, shape=torch.Size([128])
name=neck.conv.0.weight, shape=torch.Size([64, 128, 3, 3])
name=neck.conv.0.bias, shape=torch.Size([64])
	"""
	def __init__(self):
		super().__init__()
		self.upsample = nn.Sequential(
			nn.UpsamplingNearest2d(scale_factor=2), #[1,256,16,20] *2 = #[1,256,32,40]
			nn.Conv2d(256, 128, (3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True),#[1,128,32,40]
			nn.LeakyReLU(0.1, inplace=False)
		)
		self.conv = nn.Sequential(
			nn.Conv2d(128, 64, (3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True),#[1,64,32,40]
			nn.LeakyReLU(0.1, inplace=False),
		)

	def forward(self,
				x4: torch.tensor,  #[1,256,16,20]
				x3: torch.tensor): #[1,128,32,40]
		_0 = self.upsample(x4) + x3
		return self.conv(_0)

class CenterPointHead(nn.Module):
	"""
name=head.heatmap_head.0.weight, shape=torch.Size([64, 64, 3, 3])
name=head.heatmap_head.0.bias, shape=torch.Size([64])
name=head.heatmap_head.2.weight, shape=torch.Size([9, 64, 1, 1])
name=head.heatmap_head.2.bias, shape=torch.Size([9])
name=head.offset_head.0.weight, shape=torch.Size([64, 64, 3, 3])
name=head.offset_head.0.bias, shape=torch.Size([64])
name=head.offset_head.2.weight, shape=torch.Size([2, 64, 1, 1])
name=head.offset_head.2.bias, shape=torch.Size([2])
	"""
	def __init__(self):
		super().__init__()
		self.offset_head = nn.Sequential(
			nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True), #[1,64,32,40]
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 2, (1, 1), stride=1, padding=0, dilation=1, groups=1, bias=True)   #[1,2,32,40]
		)
		self.heatmap_head = nn.Sequential(
			nn.Conv2d(64, 64, (3, 3), stride=1, padding=1, dilation=1, groups=1, bias=True), #[1,64,32,40]
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 9, (1, 1), stride=1, padding=0, dilation=1, groups=1, bias=True)   #[1,9,32,40]
		)
		# self.loss_center_heatmap = GaussianFocalLoss #todo
		self.loss_offset = nn.L1Loss()

	def forward(self, x: torch.tensor):
		return torch.sigmoid(self.heatmap_head(x)), self.offset_head(x)


class Module(nn.Module):
	def __init__(self):
		super().__init__()
		self.bone = Backbone()
		self.neck = Neck()
		self.head = CenterPointHead()
		# out = F.avg_pool2d(out, 2)
		# out = out.view(out.size(0), -1)
		self.fc = nn.Linear(64, 10)
		# out = self.fc(out)

	def forward(self, x):
		x4, x3 = self.bone(x)
		_2 = self.neck(x4, x3)
		_3, _4 = self.head(_2)
		return [_3, _4]

	def forward_on_cifa(self, x):
		out = self.neck(*self.bone(x))
		out = F.avg_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = self.fc(out)
		return out

class TinyNet(Module):
	def __init__(self,device, input_size=None, trainable=False, num_classes=None, backbone='r18', conf_thresh=0.05, nms_thresh=0.45, topk=100, gs=1.0, use_nms=False):
		super().__init__()
		self.device = device
		self.input_size = input_size
		self.trainable = trainable
		self.num_classes = num_classes
		# self.bk = backbone
		# self.conf_thresh = conf_thresh
		# self.nms_thresh = nms_thresh
		# self.stride = 4
		# self.topk = topk
		# self.gs = gs
		# self.use_nms = use_nms
		# self.grid_cell = self.create_grid(input_size)
		self.cls_loss_func = HeatmapLoss(reduction='mean')
		self.txty_loss_func = nn.BCEWithLogitsLoss(reduction='none')
		self.twth_loss_func = nn.SmoothL1Loss(reduction='none')

	def forward(self, x, label=None):
		# cls_pred  # [b,80,32,32]
		# txty_pred # [b,2,32,32]

		cls_pred, txty_pred = super().forward(x)
		B = cls_pred.size(0)
		assert cls_pred.size(0) == txty_pred.size(0)
		if self.trainable:
			# [B, H*W, num_classes]
			cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)  # [b,1024,80]
			# [B, H*W, 2]
			txty_pred = txty_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2) # [b,1024,2]

			#groundtruth
			gt_cls = label[:, :, :self.num_classes]  #todo,gt的维度也要变为32*32
			gt_txty = label[:, :, self.num_classes: self.num_classes + 2]
			# gt_twth = label[:, :, num_classes + 2: num_classes + 4]
			gt_box_scale_weight = label[:, :, num_classes + 4]
			cls_loss = self.cls_loss_func(cls_pred, gt_cls)

			# box loss
			txty_loss = torch.sum(torch.sum(self.txty_loss_func(txty_pred, gt_txty), dim=-1) * gt_box_scale_weight) / B
			# twth_loss = torch.sum(torch.sum(self.twth_loss_func(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight) / B

			totalLoss = cls_loss + txty_loss
			return totalLoss
		else: #test
			with torch.no_grad():
				# batch_size = 1
				cls_pred = torch.sigmoid(cls_pred)
				# self.vis_fmap(cls_pred[0][-1], normal=True, name='cls_pred_0')

				# simple nms
				hmax = F.max_pool2d(cls_pred, kernel_size=5, padding=2, stride=1)
				keep = (hmax == cls_pred).float()
				cls_pred *= keep


def train_on_cifa10():
	""" train backbone&neck on cifa10"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	saveDir = os.path.join('./weights/')
	os.makedirs(saveDir, exist_ok=True)

	tblogger = util.getTensorBoard()
	util.startTensorboard('./log/')

	# set hyperparameter
	EPOCH = 920
	pre_epoch = 0
	BATCH_SIZE = 128
	# LR = 0.01
	LR = 0.0016

	# prepare dataset and preprocessing
	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
	])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

	testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	# labels in CIFAR10
	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	# define tiny-net
	net = Module()
	net.to(device)
	# keep training
	resume = True
	pthFile = "./weights/tinyOnCifa10_355_83.630.pth"
	if resume:
		assert os.path.exists(pthFile)
		print('keep training model: %s' % (pthFile))
		net.load_state_dict(torch.load(pthFile, map_location=device))

	# define loss funtion & optimizer
	criterion = nn.CrossEntropyLoss()
	# optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
	optimizer = optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-5)

	# train
	for epoch in range(pre_epoch, EPOCH):
		print('\nEpoch: %d' % (epoch + 1))
		net.train()
		sum_loss = 0.0
		correct = 0.0
		total = 0.0
		for i, data in enumerate(trainloader, 0):
			# prepare dataset
			length = len(trainloader)
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)
			optimizer.zero_grad()

			# forward & backward
			outputs = net.forward_on_cifa(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print ac & loss in each batch
			sum_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += predicted.eq(labels.data).cpu().sum()
			tblogger.add_scalar('train/Loss', sum_loss / (i + 1), i + epoch * length)
			tblogger.add_scalar('train/Acc', 100. * correct / total, i + epoch * length)
			# print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
			# 	  % (epoch, (i + epoch * length), sum_loss / (i + 1), 100. * correct / total))

		# get the ac with testdataset in each epoch
		print('Waiting Test...')
		with torch.no_grad():
			correct = 0
			total = 0
			for data in testloader:
				net.eval()
				images, labels = data
				images, labels = images.to(device), labels.to(device)
				outputs = net.forward_on_cifa(images)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum()
			# print('Test\'s ac is: %.3f%%' % (100 * correct / total))
			tblogger.add_scalar('test/Acc', 100. * correct / total, epoch)
			print('Saving state, epoch:', epoch)
			torch.save(net.state_dict(), os.path.join(saveDir,  'tinyOnCifa10_' + repr(epoch) + '_' +'%.3f'%(100 * correct / total) + '.pth'),
					   _use_new_zipfile_serialization=False
					   )

	print('Train has finished, total epoch is %d' % EPOCH)


def test():
	dummy_input = torch.rand(1, 3, 512, 640)

	net = Module()
	# print(a)
	print("=====")
	out = net.bone.feature_00(dummy_input)
	out = net.bone.feature_01(out)
	out = net.bone.feature_10(out)
	out = net.bone.feature_11(out)
	out = net.bone.feature_20(out)
	out = net.bone.feature_21(out)
	out = net.bone.feature_30(out)
	out = net.bone.feature_31(out)
	out = net.bone.feature_40(out)
	out = net.bone.feature_41(out)
	out = net.bone.feature_42(out)

	x4, x3 = net.bone(dummy_input)
	out_neck = net.neck(x4, x3)
	out = net.head.offset_head(out_neck)
	out = net.head.heatmap_head(out_neck)
	print(out.shape)
	print("=====")
	# print(net)
	# print(a(dummy_input))

if __name__ == '__main__':
	# test()
	train_on_cifa10()
