import os
import cv2
import json
import torch
import random
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
# from labelme import utils

g_dataset_root = '/media/e68d0b9b-67bc-4cb2-9235-69d7bae80077/home/znzz/COCO/2024_01_18_17_54_06/'

import setup
if __name__ == '__main__':
	setup.setCurPath(__file__)
from data import create_gt
import cv.cvUtility as cvUtility
import util

g_labelIndexLut = {
	'corss': 0,
	'corner': 0,
	'cross': 1,
	'dot': 2,
	'l_light': 3,
	'c_light': 4,
	'_background_': 5
}

class FileReader(cvUtility.FileReader):
	def __init__(self, image_folder, suffixs=[".png"]):
		super().__init__(image_folder, suffixs)

	def validFile(self, fileName):
		assert os.path.exists(fileName)
		label = json.load(open(fileName))
		return len(label['shapes']) > 0


class PointsDataset(Dataset):
	def __init__(self,
				 data_dir=g_dataset_root,
				 img_size=512,
				 train=False,
				 stride=16,
				 transform=util.BaseTransform([512, 512], (0, 0, 0)),
				 base_transform=util.BaseTransform([512, 512], (0, 0, 0)),
				 mosaic=False):
		"""
		COCO dataset initialization. Annotation data are read into memory by COCO API.
		Args:
			data_dir (str): dataset root directory
			json_file (str): COCO json file name
			name (str): COCO data name (e.g. 'train2017' or 'val2017')
			img_size (int): target image size after pre-processing
			min_size (int): bounding boxes smaller than this are ignored
			debug (bool): if True, only one data id is selected from the dataset
		"""
		self.data_dir = data_dir
		self.img_size = img_size
		self.train = train
		self.stride = stride
		self.name = "fastprint_knowledge_city"
		self.transform = transform
		self.base_transform = base_transform
		self.mosaic = mosaic
		self.reader = FileReader(os.path.join(data_dir,"labels"), [".json"])

	def __len__(self):
		return self.reader.max_idx

	def pull_image(self, index:int):
		label = self.reader.image_list[index]
		# print("pull idx:{},{}".format(index, label))
		# assert os.path.exists(label)
		label = json.load(open(label))
		img = cv2.imread(os.path.join(g_dataset_root, "images", label['imagePath']))
		assert img.shape[0] == label["imageHeight"]
		assert img.shape[1] == label["imageWidth"]
		assert img.shape[2] == 3
		shapes = label['shapes']
		return img

	def pull_anno(self, index:int):
		assert index <= self.reader.max_idx
		anno = self.reader.image_list[index]
		assert os.path.exists(anno)
		anno = json.load(open(anno))
		# shapes = anno['shapes']
		# lbl = utils.shapes_to_label(img.shape, anno['shapes'], label_name_to_value)
		target = []
		r = 16
		for shape in anno['shapes']:
			assert len(shape['points']) == 1
			p = shape['points'][0]
			center = (int(p[0]), int(p[1]))
			tl = (center[0] - r, center[1] - r)
			br = (center[0] + r, center[1] + r)
			xmin = max(0, tl[0])
			ymin = max(0, tl[1])
			xmax = min(br[0], anno['imageWidth'])
			ymax = min(br[1], anno['imageHeight'])
			target.append([xmin, ymin, xmax, ymax, g_labelIndexLut[shape['label']]])
		return target

	def __getitem__(self, index):
		im, gt, h, w = self.pull_item(index)
		if self.train:
			gt_tensor = create_gt.gt_creator(img_size=self.img_size,
											 stride=self.stride,
											 num_classes=80,
											 label_lists=gt)
			return im, gt_tensor
		else:
			return im, gt

	def pull_item(self, index:int):
		"""
		:return: image target h w;
		target=[[xmin, ymin, xmax, ymax, cls_id]]
		"""
		img = self.pull_image(index)
		# targets = self.pull_anno(index)
		assert img is not None

		height, width, channels = img.shape

		# COCOAnnotation Transform
		# start here :
		target = []
		for anno in self.pull_anno(index):
			xmin, ymin, xmax, ymax, clsId = anno
			xmin /= width
			ymin /= height
			xmax /= width
			ymax /= height
			target.append([xmin, ymin, xmax, ymax, clsId])
		# end here .

		# mosaic augmentation
		# if self.mosaic and np.random.randint(2):
		if self.mosaic:
			# ids_list_ = self.ids[:index] + self.ids[index + 1:]
			ids_list_ = [i for i in range(0,index)] + [i for i in range(index+1,self.reader.max_idx+1)] #[i.alive() for i in cstats]
			# random sample 3 indexs
			id2, id3, id4 = random.sample(ids_list_, 3)
			ids = [id2, id3, id4]
			img_lists = [img]
			tg_lists = [target]
			# load other 3 images and targets
			for id_ in ids:
				assert id_ <= self.reader.max_idx
				img_i = self.pull_image(id_)
				assert img_i is not None

				height_i, width_i, channels_i = img_i.shape
				# COCOAnnotation Transform
				# start here :
				target_i = []
				for anno in self.pull_anno(index):
					xmin, ymin, xmax, ymax, clsId = anno
					xmin /= width
					ymin /= height
					xmax /= width
					ymax /= height
					target_i.append([xmin, ymin, xmax, ymax, clsId])
				# end here .
				img_lists.append(img_i)
				tg_lists.append(target_i)

			mosaic_img = np.zeros([self.img_size * 2, self.img_size * 2, img.shape[2]], dtype=np.uint8)
			# mosaic center
			yc, xc = [int(random.uniform(-x, 2 * self.img_size + x)) for x in
					  [-self.img_size // 2, -self.img_size // 2]]
			# yc = xc = self.img_size

			mosaic_tg = []
			for i in range(4):
				img_i, target_i = img_lists[i], tg_lists[i]
				h0, w0, _ = img_i.shape

				# resize image to img_size
				img_i = cv2.resize(img_i, (self.img_size, self.img_size))
				h, w, _ = img_i.shape

				# place img in img4
				if i == 0:  # top left
					x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
					x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
				elif i == 1:  # top right
					x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
					x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
				elif i == 2:  # bottom left
					x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
					x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
				elif i == 3:  # bottom right
					x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
					x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

				mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
				padw = x1a - x1b
				padh = y1a - y1b

				# labels
				target_i = np.array(target_i)
				target_i_ = target_i.copy()
				if len(target_i) > 0:
					# a valid target, and modify it.
					target_i_[:, 0] = (w * (target_i[:, 0]) + padw)
					target_i_[:, 1] = (h * (target_i[:, 1]) + padh)
					target_i_[:, 2] = (w * (target_i[:, 2]) + padw)
					target_i_[:, 3] = (h * (target_i[:, 3]) + padh)

					mosaic_tg.append(target_i_)

			if len(mosaic_tg) == 0:
				mosaic_tg = np.zeros([1, 5])
			else:
				mosaic_tg = np.concatenate(mosaic_tg, axis=0)
				# Cutout/Clip targets
				np.clip(mosaic_tg[:, :4], 0, 2 * self.img_size, out=mosaic_tg[:, :4])
				# normalize
				mosaic_tg[:, :4] /= (self.img_size * 2)

			# augment
			mosaic_img, boxes, labels = self.base_transform(mosaic_img, mosaic_tg[:, :4], mosaic_tg[:, 4])
			# to rgb
			mosaic_img = mosaic_img[:, :, (2, 1, 0)]
			mosaic_tg = np.hstack((boxes, np.expand_dims(labels, axis=1)))

			return torch.from_numpy(mosaic_img).permute(2, 0, 1).float(), mosaic_tg, self.img_size, self.img_size

		# basic augmentation(SSDAugmentation or BaseTransform)
		else:
			# check targets
			if len(target) == 0:
				target = np.zeros([1, 5])
			else:
				target = np.array(target)

			img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
			# to rgb
			img = img[:, :, (2, 1, 0)]
			target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

			return torch.from_numpy(img).permute(2, 0, 1), target, height, width


class Evaluator(object):
	def __init__(self, train=True, stride=4, device='cpu'):
		self.dataset = PointsDataset(train=train, stride=stride)
		self.device = device

	def evaluate(self, net):
		net.eval()
		for i in range(len(self.dataset)):
			im, gt, h, w = self.dataset.pull_item(i)

			x = Variable(im.unsqueeze(0)).to(self.device)
			net(x)

			im = im.permute(1, 2, 0) / 255.
			# im = im.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
			# im = im.copy()
			# cv2.imshow("im", im)
			# cv2.waitKey(500)
			# plt.figure("im")
			# plt.imshow(im)

			# plt.show()
			# assert False
			# bboxes, scores, cls_inds = net(x)
			# scale = np.array([[w, h, w, h]])
			# bboxes *= scale

#########################################################################################
def test_evaluator():
	device = 'cpu'
	from models.centernet_plus_simple import CenterNetPlus
	net = CenterNetPlus(device=device,
					input_size=512,
					num_classes=80,
					trainable=False,
					backbone='r18')

	resume = '../weights/fastprint/centernet_plus/centernet_plus_18.pth'
	print('load model: %s' % (resume))
	assert os.path.exists(resume)
	net.load_state_dict(torch.load(resume, map_location=device))
	eva = Evaluator(device=device)
	eva.evaluate(net)

def test_image_viewer():
	cap = cvUtility.ImageReader(util.g_nn_data, [".png"])
	plt.figure()
	while True:
		ret, mat = cap.read()
		assert ret
		# plt.imshow(mat)
		# plt.pause(2)
		# plt.show()
		cvUtility.show("mat", cv2.resize(mat,(1024,1024), interpolation=cv2.INTER_NEAREST))
		cvUtility.wait()

def test():
	global g_dataset_root
	json_path = os.path.join(g_dataset_root, 'labels/1.json')
	print("====", json_path)
	assert os.path.exists(json_path)
	data = json.load(open(json_path))
	img = cv2.imread(os.path.join(g_dataset_root, "images", data['imagePath']))
	assert img.shape[0] == data["imageHeight"]
	assert img.shape[1] == data["imageWidth"]
	assert img.shape[2] == 3
	shapes = data['shapes']
	for s in shapes:
		print(s['label'])
		print(s['points'])
		assert len(s['points']) == 1
		p = s['points'][0]
		center = (int(p[0]), int(p[1]))
		tl = (center[0] - 5, center[1] - 5)
		br = (center[0] + 5, center[1] + 5)
		cv2.rectangle(img, tl, br, (0,255,0), 1, cv2.LINE_AA)
		cv2.circle(img, center, 1, (0,0,255), -1)
		cv2.circle(img, ((tl[0] + br[0])//2,(tl[1] + br[1])//2) , 1, (255,0,0), -1)
	cv2.imshow("1", img)
	cv2.waitKey()

def test_PointsDataset():
	dataset = PointsDataset(stride=1)
	ids_list_ = [i for i in range(0, dataset.reader.max_idx+1)]
	id = random.sample(ids_list_, 1)
	# im, gt = dataset[id[0]]
	im, gt = dataset[2015]
	im = im.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
	im = im.copy()
	for box in gt:
		print("==", box)
		xmin, ymin, xmax, ymax, _ = box
		xmin *= im.shape[1]
		ymin *= im.shape[0]
		xmax *= im.shape[1]
		ymax *= im.shape[0]
		im = cv2.rectangle(im, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 1)
	cv2.imshow('gt', im)
	cv2.waitKey()

def test_train_PointsDataset():
	stride = 16
	dataset = PointsDataset(train=True,stride=stride,mosaic=False)
	ids_list_ = [i for i in range(0, dataset.reader.max_idx+1)]
	id = random.sample(ids_list_, 1)
	im, gt = dataset[id[0]]
	im = im.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)].astype(np.uint8)
	im = im.copy()
	plt.figure("img")
	plt.imshow(im)
	print("im.shape=", im.shape)
	print("gt.shape=", gt.shape)
	for i in range(gt.shape[-1]):
		hm = gt[:,i].reshape(-1, im.shape[1]//stride).cpu().numpy()
		plt.figure(i)
		plt.imshow(hm)
		plt.show()

if __name__ == '__main__':
	# test_PointsDataset()
	# test_train_PointsDataset()
	test_evaluator()
	# test_image_viewer()

