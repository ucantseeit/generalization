import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

# Define LeNet model
class LeNet(nn.Module):
	def __init__(self, num_classes=10, in_channels=1): # Default to 1 channel for MNIST
		super(LeNet, self).__init__()
		# First convolutional layer
		# Output size after conv1 (kernel 5, no padding, stride 1): (Input_H - 5 + 1) = Input_H - 4
		# For 32x32 input -> 28x28
		self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
		self.relu1 = nn.ReLU()
		# Output size after pool1 (kernel 2, stride 2): Input_H / 2
		# For 28x28 input -> 14x14
		self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

		# Second convolutional layer
		# Output size after conv2: (Input_H - 5 + 1) = Input_H - 4
		# For 14x14 input -> 10x10
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
		self.relu2 = nn.ReLU()
		# Output size after pool2: Input_H / 2
		# For 10x10 input -> 5x5
		self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

		# Third convolutional layer (acting as fully connected to previous layer in original LeNet-5)
		# For 5x5 input, conv3 (kernel 5, no padding, stride 1) -> 1x1 output
		self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
		self.relu3 = nn.ReLU()

		# Fully connected layers
		# The input size to fc1 is 120 because conv3 (C5 layer) outputs 120 feature maps,
		# each of size 1x1 (when the input to C5 is 5x5 as from S4 layer).
		self.fc1 = nn.Linear(120, 84)
		self.relu4 = nn.ReLU()
		self.fc2 = nn.Linear(84, num_classes)

	def forward(self, x):
		x = self.pool1(self.relu1(self.conv1(x)))
		x = self.pool2(self.relu2(self.conv2(x)))
		x = self.relu3(self.conv3(x))
		# Flatten the output of the last convolutional layer (conv3)
		# Shape after conv3 is (batch_size, 120, 1, 1), flatten to (batch_size, 120)
		x = torch.flatten(x, 1) # Flatten starting from dimension 1 (batch_size, channels*H*W)
		x = self.relu4(self.fc1(x))
		x = self.fc2(x)
		return x



# github https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
	classname = m.__class__.__name__
	#print(classname)
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
		init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
	def __init__(self, lambd):
		super(LambdaLayer, self).__init__()
		self.lambd = lambd

	def forward(self, x):
		return self.lambd(x)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1, option='A'):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != planes:
			if option == 'A':
				"""
				For CIFAR10 ResNet paper uses option A.
				"""
				self.shortcut = LambdaLayer(lambda x:
											F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
			elif option == 'B':
				self.shortcut = nn.Sequential(
					 nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
					 nn.BatchNorm2d(self.expansion * planes)
				)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 16
		
		self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(16)
		self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
		self.linear = nn.Linear(64, num_classes)

		self.apply(_weights_init)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion

		return nn.Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = F.avg_pool2d(out, out.size()[3])
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out


def resnet20():
	return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
	return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
	return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
	return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
	return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
	return ResNet(BasicBlock, [200, 200, 200])


def get_model(dataset_name : str, model_name : str, weights, num_classes):
	dataset_name = dataset_name.lower()
	model_name = model_name.lower()
	if dataset_name == 'imagenet' or dataset_name == 'tinyimagenet':
		return torchvision.models.get_model(model_name, weights=weights, num_classes=num_classes) 
	elif dataset_name == 'cifar10' and model_name == 'resnet20':
		return resnet20()
	elif dataset_name == 'mnist' and model_name == 'lenet':
		return LeNet()
	else:
		return ValueError("The model for this dataset is not defined, please check")