import time
import os

import pytorch_script.presets as presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms

from pytorch_script.sampler import RASampler
from torchvision.transforms.functional import InterpolationMode

import pytorch_script.utils as utils

def _get_cache_path(filepath):
	import hashlib

	h = hashlib.sha1(filepath.encode()).hexdigest()
	# cache_path = os.path.join("~", ".torch", "vision", "datasets", 
	# 					   "imagefolder", h[:10] + ".pt")
	cache_path = os.path.join("~", ".torch", "vision", "datasets", 
							f"{os.path.basename(filepath.strip('/'))}-{h[:10]}.pt")
	cache_path = os.path.expanduser(cache_path)
	return cache_path

def load_imagenet_data(args):
	# Data loading code
	print(f"Loading ImageNet data from {args.data_path}")

	# ImageNet 的数据路径假设包含 train 和 val 子目录
	train_data_root = os.path.join(args.data_path, "train")
	val_data_root = os.path.join(args.data_path, "val")

	# 读取prests参数
	interpolation = InterpolationMode(args.interpolation)
	backend = args.backend
	val_resize_size, val_crop_size, train_crop_size = (
		args.val_resize_size,
		args.val_crop_size,
		args.train_crop_size,
	)

	print("Loading training data")
	st = time.time()
	cache_path = _get_cache_path(train_data_root)
	if args.cache_dataset and os.path.exists(cache_path):
		# Attention, as the transforms are also cached!
		print(f"Loading dataset_train from {cache_path}")
		# TODO: this could probably be weights_only=True
		dataset, _ = torch.load(cache_path, weights_only=False)
	else:
		# We need a default value for the variables below because args may come
		# from train_quantization.py which doesn't define them.
		auto_augment_policy = getattr(args, "auto_augment", None)
		random_erase_prob = getattr(args, "random_erase", 0.0)
		ra_magnitude = getattr(args, "ra_magnitude", None)
		augmix_severity = getattr(args, "augmix_severity", None)
		dataset = torchvision.datasets.ImageFolder(
			train_data_root,
			presets.ClassificationPresetTrain(
				crop_size=train_crop_size,
				interpolation=interpolation,
				auto_augment_policy=auto_augment_policy,
				random_erase_prob=random_erase_prob,
				ra_magnitude=ra_magnitude,
				augmix_severity=augmix_severity,
				backend=backend,
			),
		)
		if args.cache_dataset:
			print(f"Saving dataset_train to {cache_path}")
			utils.mkdir(os.path.dirname(cache_path))
			utils.save_on_master((dataset, train_data_root), cache_path)
	print("Took", time.time() - st)

	print("Loading validation data")
	cache_path = _get_cache_path(val_data_root)
	if args.cache_dataset and os.path.exists(cache_path):
		# Attention, as the transforms are also cached!
		print(f"Loading dataset_test from {cache_path}")
		# TODO: this could probably be weights_only=True
		dataset_test, _ = torch.load(cache_path, weights_only=False)
	else:
		if args.weights and args.test_only:
			weights = torchvision.models.get_weight(args.weights)
			preprocessing = weights.transforms(antialias=True)
			if args.backend == "tensor":
				preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

		else:
			preprocessing = presets.ClassificationPresetEval(
				crop_size=val_crop_size,
				resize_size=val_resize_size,
				interpolation=interpolation,
				backend=backend,
			)

		dataset_test = torchvision.datasets.ImageFolder(
			val_data_root,
			preprocessing,
		)
		if args.cache_dataset:
			print(f"Saving dataset_test to {cache_path}")
			utils.mkdir(os.path.dirname(cache_path))
			utils.save_on_master((dataset_test, val_data_root), cache_path)

	print("Creating data loaders")
	if args.distributed:
		if hasattr(args, "ra_sampler") and args.ra_sampler:
			train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
		else:
			train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
		test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
	else:
		train_sampler = torch.utils.data.RandomSampler(dataset)
		test_sampler = torch.utils.data.SequentialSampler(dataset_test)

	return dataset, dataset_test, train_sampler, test_sampler


# 不需要加载缓存，因为数据集较小
def load_cifar10_data(args):
	print(f"Loading CIFAR10 data from {args.data_path}")

	st = time.time()
	print("Start Downloading CIFAR10")
	dataset_train = torchvision.datasets.CIFAR10(
		root=args.data_path,
		train=True,
		download=True,
		transform=presets.CIFAR10PresetTrain(),
	)
	print("Took", time.time() - st)

	st = time.time()
	dataset_test = torchvision.datasets.CIFAR10(
		root=args.data_path,
		train=False,
		download=True,
		transform=presets.CIFAR10PresetEval(),
	)
	print("Took", time.time() - st)
	num_classes = 10

	return dataset_train, dataset_test, num_classes


def load_mnist_data(args):
	print(f"Loading MNIST data from {args.data_path}")

	st = time.time()
	print("Start Downloading MNIST train dataset")
	dataset_train = torchvision.datasets.MNIST(
		root=args.data_path,
		train=True,
		download=True, # Ensure download is set to True
		transform=presets.MNISTPresetTrain(),
	)
	print("Took", time.time() - st)
	
	st = time.time()
	print("Start Downloading MNIST test dataset")
	dataset_test = torchvision.datasets.MNIST(
		root=args.data_path,
		train=False,
		download=True,
		transform=presets.MNISTPresetEval(),
	)
	print("Took", time.time() - st)
	num_classes = 10 # MNIST has 10 classes

	return dataset_train, dataset_test, num_classes

from PIL import Image
from torch.utils.data import Dataset

class TinyImageNetValDataset(Dataset):
	def __init__(self, root_dir, transform=None):
		"""
		Tiny ImageNet 验证集 Dataset 类。

		Args:
			root_dir (str): Tiny ImageNet 数据集根目录的路径，
							例如 'path/to/tiny-imagenet-200/val'。
			transform (callable, optional): 应用于图像的转换。
		"""
		self.root_dir = root_dir
		self.transform = transform
		self.annotations_file = os.path.join(root_dir, 'val_annotations.txt')
		self.image_paths = []
		self.labels = []
		self.class_to_idx = {} # 用于将类别名映射到整数索引

		self._load_annotations()

	def _load_annotations(self):
		"""
		解析 val_annotations.txt 文件并构建图像路径和标签列表。
		由此得到 self.image_paths 与 self.labels, 其顺序与anntatoins.txt中一致
		还构建了self.class_to_idx, 这是用来确保与训练集class的命名一致
		"""
		unique_classes = set()
		with open(self.annotations_file, 'r') as f:
			for line in f:
				parts = line.strip().split('\t')
				if len(parts) >= 2:
					image_filename = parts[0]
					class_name = parts[1]
					unique_classes.add(class_name)

		# 对类别名进行排序，以确保索引的一致性
		sorted_classes = sorted(list(unique_classes))
		self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted_classes)}

		# 现在再次遍历注解文件，填充 image_paths 和 labels
		with open(self.annotations_file, 'r') as f:
			for line in f:
				parts = line.strip().split('\t')
				image_filename = parts[0]
				class_name = parts[1]
				
				image_path = os.path.join(self.root_dir, 'images', image_filename)
				if os.path.exists(image_path): # 检查文件是否存在
					self.image_paths.append(image_path)
					self.labels.append(self.class_to_idx[class_name])
						
		print(f"Found {len(self.image_paths)} images in the validation set.")


	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, sample_idx):
		img_path = self.image_paths[sample_idx]
		label = self.labels[sample_idx]

		image = Image.open(img_path).convert('RGB')
		if self.transform:
			image = self.transform(image)

		return image, label

def load_tinyimagenet_data(data_path):
	if not os.path.exists(data_path):
		raise FileNotFoundError(f"Tiny ImageNet dataset not found at {data_path}. Please download it manually.")

	train_dir = os.path.join(data_path, 'train')
	# 验证集路径 (通常是 val/ 目录)
	# 注意：Tiny ImageNet的val目录需要特殊处理，因为其子文件夹不是类别，而是'images'
	# 实际使用时，您可能需要一个脚本来将val/images下的图片根据val_annotations.txt组织成类别文件夹
	val_dir = os.path.join(data_path, 'val')

	if not os.path.exists(train_dir) or not os.path.exists(val_dir):
		raise FileNotFoundError(f"Tiny ImageNet train or val directories not found under {data_path}.")
	
	transform_train = presets.TinyImageNetPresetTrain()
	transform_test = presets.TinyImageNetPresetEval()

	train_dataset = torchvision.datasets.ImageFolder(
		train_dir,
		transform=transform_train
	)
	test_dataset = TinyImageNetValDataset(val_dir, transform=transform_test)

	num_classes = len(train_dataset.classes) if train_dataset.classes else 200 # Tiny ImageNet有200个类别

	return train_dataset, test_dataset, num_classes

def load_data(args):
	dataset_train = None
	dataset_test = None
	num_classes = 0

	if args.dataset_name.lower() == "imagenet":
		dataset_train, dataset_test, num_classes = load_imagenet_data(args)
	elif args.dataset_name.lower() == "cifar10":
		dataset_train, dataset_test, num_classes = load_cifar10_data(args)
	elif args.dataset_name.lower() == "mnist":
		dataset_train, dataset_test, num_classes = load_mnist_data(args)
	elif args.dataset_name == 'tinyimagenet': # 添加这一行
		dataset_train, dataset_test, num_classes = load_tinyimagenet_data(args.data_path)
	else:
		raise ValueError(f"Unsupported dataset: {args.dataset_name}. Please choose from 'imagenet', 'cifar10'.")
	
	# 采样器逻辑保持不变，因为它们通常与数据集类型无关，只与分布式设置有关
	if args.distributed:
		# 仅对 ImageNet 启用 RA Sampler，或者根据需要调整逻辑
		if hasattr(args, "ra_sampler") and args.ra_sampler and args.dataset_name.lower() == "imagenet":
			train_sampler = RASampler(dataset_train, shuffle=True, repetitions=args.ra_reps)
		else:
			train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
		test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
	else:
		train_sampler = torch.utils.data.RandomSampler(dataset_train)
		test_sampler = torch.utils.data.SequentialSampler(dataset_test)

	return dataset_train, dataset_test, train_sampler, test_sampler, num_classes