#https://www.kaggle.com/code/marquis03/shufflenet-v2-cifar-10-classification
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns #pip install seaborn

sns.set_theme(style="darkgrid", font_scale=1.5, font="SimHei", rc={"axes.unicode_minus":False})

import torch
import torchmetrics #pip install torchmetrics
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models

import lightning.pytorch as pl #pip install lightning
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
pl.seed_everything(seed)

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
])

test_transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root="./data", train=True, transform=train_transform, download=True)
val_dataset = datasets.CIFAR10(root="./data", train=False, transform=test_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = val_loader

#visualization
class_names = train_dataset.classes
class_count = [train_dataset.targets.count(i) for i in range(len(class_names))]
df = pd.DataFrame({"Class": class_names, "Count": class_count})

plt.figure(figsize=(12, 8), dpi=100)
sns.barplot(x="Count", y="Class", data=df)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 20), dpi=100)
images, labels = next(iter(val_loader))
for i in range(8):
	ax = plt.subplot(8, 4, i + 1)
	plt.imshow(images[i].permute(1, 2, 0).numpy())
	plt.title(class_names[labels[i]])
	plt.axis("off")
plt.tight_layout()
plt.show()


#modeling
class LitModel(pl.LightningModule):
	def __init__(self, num_classes=1000):
		super().__init__()
		# self.model = models.shufflenet_v2_x2_0(weights="IMAGENET1K_V1")
		self.model = models.shufflenet_v2_x2_0()
		#         for param in self.model.parameters():
		#             param.requires_grad = False
		self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
		self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)

	def forward(self, x):
		x = self.model(x)
		return x

	def configure_optimizers(self):
		optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-08, weight_decay=1e-5)
		return optimizer

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		acc = self.accuracy(y_hat, y)
		self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
		self.log('train_acc', acc, on_step=True, on_epoch=False, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		acc = self.accuracy(y_hat, y)
		self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True)
		self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

	def test_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		acc = self.accuracy(y_hat, y)
		self.log('test_acc', acc)

	def predict_step(self, batch, batch_idx, dataloader_idx=None):
		x, y = batch
		y_hat = self(x)
		preds = torch.argmax(y_hat, dim=1)
		return preds

num_classes = len(class_names)
model = LitModel(num_classes=num_classes)
model.to(device)

logger = CSVLogger("./")
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
trainer = pl.Trainer(max_epochs=20, enable_progress_bar=True, logger=logger, callbacks=[early_stop_callback],
					 accelerator="gpu")
trainer.fit(model, train_loader, val_loader)

trainer.test(model, val_loader)

#predict test data
pred = trainer.predict(model, test_loader)
pred = torch.cat(pred, dim=0)
pred = pd.DataFrame(pred.numpy(), columns=["Class"])
pred["Class"] = pred["Class"].apply(lambda x: class_names[x])

plt.figure(figsize=(12, 8), dpi=100)
sns.countplot(y="Class", data=pred)
plt.tight_layout()
plt.show()

#losss & accuracy
log_path = logger.log_dir + "/metrics.csv"
metrics = pd.read_csv(log_path)

plt.figure(figsize=(12, 8), dpi=100)
sns.lineplot(x="epoch", y="train_loss", data=metrics, label="Train Loss", linewidth=2)
sns.lineplot(x="epoch", y="train_acc", data=metrics, label="Train Accuracy", linewidth=2)
sns.lineplot(x="epoch", y="val_loss", data=metrics, label="Valid Loss", linewidth=2)
sns.lineplot(x="epoch", y="val_acc", data=metrics, label="Valid Accuracy", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.tight_layout()
plt.show()