from typing import Dict
from torch.func import vjp
import torch
from torch.func import jacrev, functional_call
import torch.nn as nn
from torch import Tensor

import heapq
from tqdm import tqdm
from torch.utils.data import DataLoader

import torchvision.transforms as T
import numpy as np
from PIL import Image

def compute_param_gradients(model: nn.Module,
									  inputs: torch.Tensor,
									  labels: torch.Tensor,
									  loss_fn = nn.CrossEntropyLoss(reduction='none'),
									  device='cuda'):
	"""
	Args:
		model: PyTorch 模型
		inputs: 输入图片，shape=(B,H,W)
		labels: 标签
		loss_fn: 损失函数, 需要reduction='none'
	
	Returns:
		grads: 
	"""
	model.eval()
	model.to(device)
	inputs, labels = inputs.to(device), labels.to(device)

	params = dict(model.named_parameters())
	buffers = dict(model.named_buffers())

	def compute_losses(params_, inputs_, labels_):
		outputs = functional_call(model, (params_, buffers), inputs_)
		losses = loss_fn(outputs, labels_)
		return losses

	grads_per_sample = jacrev(compute_losses, argnums=0)(params, inputs, labels)

	return grads_per_sample


def flatten_grads_dict(grads : dict):
	return torch.cat(tuple(grad.flatten(1) for grad in grads.values()), dim=1)


@torch.no_grad()
def get_topk_loss_sample(model, ds, loss_fn, k, batch_size=512):
	dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
	idx_offset = 0

	model.eval()
	model.to('cuda')
	heap = []
	for inputs, labels in tqdm(dl):
		inputs = inputs.to('cuda')
		labels = labels.to('cuda')

		outputs = model(inputs)
		losses = loss_fn(outputs, labels)

		batch_losses = [(loss.item(), idx_offset + j) for j, loss in enumerate(losses)]
		for loss, idx in batch_losses:
			if len(heap) < k:
				heapq.heappush(heap, (loss, idx))
			else:
				heapq.heappushpop(heap, (loss, idx))
		
		idx_offset += len(inputs)

	return sorted(heap, key=lambda x: x[0], reverse=True)


def inverse_trans_cifar10(tensor_img : Tensor) -> Image:
	tensor_img.to('cuda')
	mean = torch.tensor([0.4914, 0.4822, 0.4465], device='cuda').view(3, 1, 1)
	std = torch.tensor([0.2023, 0.1994, 0.2010], device='cuda').view(3, 1, 1)
	tensor_img = tensor_img * std + mean
	
	# 转成numpy数组，shape (H,W,C)
	np_img = tensor_img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
	# 转PIL
	pil_img = Image.fromarray(np_img)
	return pil_img


def img_show(img, ax):
	img_np = np.array(img)
	ax.imshow(img_np)
	ax.axis('off')
	return ax


def find_topk_similar_grad_samples(
	model, 
	dataset, 
	loss_fn, 
	ref_sample_idx, k, device='cuda', batch_size=128
):
	
	model.eval()
	loss_fn = loss_fn if loss_fn.reduction == 'none' else type(loss_fn)(reduction='none')

	# 计算参考样本梯度
	input_ref, label_ref = dataset[ref_sample_idx]
	input_ref = input_ref.to(device).unsqueeze(0)
	label_ref = torch.tensor(label_ref).to(device).unsqueeze(0)

	g_ref_dict = compute_param_gradients(model, input_ref, label_ref, loss_fn, device)
	g_ref = flatten_grads_dict(g_ref_dict)[0]  # shape: [D]

	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

	heap = []
	idx_offset = 0
	for inputs, labels in tqdm(dataloader):
		inputs = inputs.to(device)
		labels = labels.to(device)

		grads_dict = compute_param_gradients(model, inputs, labels, loss_fn, device)
		grads_flat = flatten_grads_dict(grads_dict)  # [B, D]

		# 计算与参考样本梯度的内积，shape [B]
		grad_inner_prods = grads_flat @ g_ref

		batch_inner_prods = [(prod.item(), idx_offset+i) \
					   for i, prod in enumerate(grad_inner_prods)]

		for prod, idx in batch_inner_prods:
			if len(heap) < k:
				heapq.heappush(heap, (prod, idx))
			else:
				heapq.heappushpop(heap, (prod, idx))
		
		idx_offset += len(inputs)

	heap.sort(key=lambda x: x[0], reverse=True)

	return heap
