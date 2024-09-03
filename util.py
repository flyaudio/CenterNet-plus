import os
import cv2
import time
import numpy as np

g_tb_logger = None
def getTensorBoard(suffix: str=""):
	from torch.utils.tensorboard import SummaryWriter
	c_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
	log_path = os.path.join('log/', suffix, c_time)
	os.makedirs(log_path, exist_ok=True)
	g_tb_logger = SummaryWriter(log_path)
	return g_tb_logger

g_tensorboard_thread = None
def startTensorboard(folder):
	import threading
	def impl(folder):
		import subprocess
		subprocess.run(f"tensorboard --logdir={str(folder)} --bind_all", shell=True)
	g_tensorboard_thread = threading.Thread(target=impl, args=(folder,))
	g_tensorboard_thread.start()

def test():
	import torch
	cls_pred = torch.rand((2, 5, 4))
	print(cls_pred)
	print(cls_pred.shape)
	# values, indices = cls_pred.topk(1, dim=0, largest=True, sorted=True)
	values, indices = torch.topk(cls_pred,1)
	print(values)


def base_transform(image, size, mean):
	x = cv2.resize(image, (size[1], size[0])).astype(np.float32)
	x -= mean
	x = x.astype(np.float32)
	return x


class BaseTransform:
	def __init__(self, size, mean):
		self.size = size
		self.mean = np.array(mean, dtype=np.float32)

	def __call__(self, image, boxes=None, labels=None):
		return base_transform(image, self.size, self.mean), boxes, labels


if __name__ == '__main__':
	test()