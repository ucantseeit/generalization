from typing import Dict
from torch.func import vjp
import torch
from torch.func import jacrev, functional_call
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

import heapq
from tqdm import tqdm
from torch.utils.data import DataLoader

import torchvision.transforms as T
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import gc, sys, traceback



def recover_cuda_after_oom():
	# 1️⃣ 清掉异常栈引用
	sys.last_traceback = None
	sys.last_value = None
	sys.last_type = None
	
	# 2️⃣ 清掉交互式变量（等价于清 _ 和 __ 的引用）
	for name in ["_", "__"]:
		if name in globals():
			globals()[name] = None
	
	# 3️⃣ 回收 Python 层的对象
	print(gc.collect())
	
	# 4️⃣ 清空 PyTorch CUDA 缓存
	torch.cuda.empty_cache()
	torch.cuda.ipc_collect()
	
	print("✅ CUDA 显存已尝试回收")



# def compute_param_gradients(model: nn.Module,
# 									  inputs: torch.Tensor,
# 									  labels: torch.Tensor,
# 									  loss_fn = nn.CrossEntropyLoss(reduction='none'),
# 									  device='cuda'):
# 	"""
# 	Args:
# 		model: PyTorch 模型
# 		inputs: 输入图片，shape=(B,H,W)
# 		labels: 标签
# 		loss_fn: 损失函数, 需要reduction='none'
	
# 	Returns:
# 		grads: 
# 	"""
# 	model.eval()
# 	model.to(device)
# 	inputs, labels = inputs.to(device), labels.to(device)

# 	params = dict(model.named_parameters())
# 	buffers = dict(model.named_buffers())

# 	def compute_losses(params_, inputs_, labels_):
# 		outputs = functional_call(model, (params_, buffers), inputs_)
# 		losses = loss_fn(outputs, labels_)
# 		return losses

# 	grads_per_sample = jacrev(compute_losses, argnums=0)(params, inputs, labels)

# 	return grads_per_sample


# def compute_param_grads(model: nn.Module, 
# 						inputs: torch.Tensor,
# 						labels: torch.Tensor,
# 						loss_fn = nn.CrossEntropyLoss(reduction='none'),
# 						device='cuda'):
# 	"""
# 	使用 torch.func.vmap 向量化地获取所有样本的梯度。
# 	"""
# 	model.to(device)
# 	model.eval()

# 	params = dict(model.named_parameters())
# 	buffers = dict(model.named_buffers())

# 	# 将 compute_single_gradient 定义为内部函数，以便访问 model
# 	def compute_single_gradient(params, buffers, input, label):
# 		outputs = functional_call(model, (params, buffers), input.unsqueeze(0))
# 		losses = loss_fn(outputs, label.unsqueeze(0))
# 		return losses

# 	# vmap 向量化 jacrev，用于批量计算
# 	grads_fn = jacrev(compute_single_gradient, argnums=0)
# 	vmap_grads_fn = torch.func.vmap(grads_fn, in_dims=(None, None, 0, 0))

# 	inputs, labels = inputs.to(device), labels.to(device)
	
# 	# 使用 vmap_grads_fn 进行高效的批量梯度计算
# 	grads_per_sample_dict = vmap_grads_fn(params, buffers, inputs, labels)
	
# 	flattened_grad = flatten_grads_dict(grads_per_sample_dict)
	
# 	# 释放缓存以防万一
# 	torch.cuda.empty_cache()

# 	return flattened_grad

def compute_param_grads(model: nn.Module, 
						inputs: torch.Tensor,
						labels: torch.Tensor,
						loss_fn = nn.CrossEntropyLoss(reduction='mean'),
						device='cuda'):
	"""
	使用 torch.func.vmap 向量化地获取所有样本的梯度。
	"""
	loss_fn = loss_fn if loss_fn.reduction == 'mean' else type(loss_fn)(reduction='mean')

	model.to(device)
	model.eval()

	params = dict(model.named_parameters())
	buffers = dict(model.named_buffers())

	# 将 compute_single_loss 定义为内部函数，以便访问 model
	def compute_single_loss(params, buffers, input, label):
		output = functional_call(model, (params, buffers), input.unsqueeze(0))
		loss = loss_fn(output, label.unsqueeze(0))
		return loss

	# vmap 向量化 jacrev，用于批量计算
	grads_fn = torch.func.grad(compute_single_loss, argnums=0)
	vmap_grads_fn = torch.func.vmap(grads_fn, in_dims=(None, None, 0, 0))

	inputs, labels = inputs.to(device), labels.to(device)
	
	# 使用 vmap_grads_fn 进行高效的批量梯度计算
	grads_per_sample_dict = vmap_grads_fn(params, buffers, inputs, labels)
	
	flattened_grad = flatten_grads_dict(grads_per_sample_dict)
	
	# 释放缓存以防万一
	torch.cuda.empty_cache()

	return flattened_grad


def flatten_grads_dict(grads : dict):
	return torch.cat(tuple(grad.flatten(1) for grad in grads.values()), dim=1).detach().requires_grad_(False)

def inverse_trans_cifar10(tensor_img):
	mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
	
	for t, m, s in zip(tensor_img, mean, std):
		t.mul_(s).add_(m)
	tensor_img = torch.clamp(tensor_img, 0, 1)
	
	# 转成numpy数组，shape (H,W,C)
	np_img = tensor_img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
	# 转PIL
	pil_img = Image.fromarray(np_img)
	return pil_img

def img_show(img, ax=None):
	if ax is None:
		fig, ax = plt.subplots()
	img_np = np.array(img)
	ax.imshow(img_np)
	ax.axis('off')
	return ax



def find_topk_samples(
	ds, 
	fn,
	k=16,
	batch_size=256,
	reverse=False,
	show_progress=True
):
	
	dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

	heap = []
	idx_offset = 0

	iterator = tqdm(dl) if show_progress else dl
	for batch in iterator:

		batch_res = fn(batch)
		if reverse:
			# 取最小k个，先取负数，再存进堆
			batch_res_idx = [(-res, idx_offset+i) for i, res in enumerate(batch_res)]
		else:
			batch_res_idx = [(res, idx_offset+i) for i, res in enumerate(batch_res)]

		for res_idx in batch_res_idx:
			if len(heap) < k:
				heapq.heappush(heap, res_idx)
			else:
				heapq.heappushpop(heap, res_idx)
		
		idx_offset += len(batch[0])

	if reverse:
		# 负数还原
		heap = [(-res, idx) for res, idx in heap]

	heap.sort(key=lambda x: x[0], reverse=not reverse)

	return heap

@torch.no_grad()
def get_batch_loss_fn(model, loss_fn=nn.CrossEntropyLoss(reduction='none'), device='cuda'):
	model.eval()
	loss_fn = loss_fn if loss_fn.reduction == 'none' else type(loss_fn)(reduction='none')
	model = model.to(device)
	def f(batch):
		inputs, labels = batch
		inputs, labels = inputs.to(device), labels.to(device)
		outputs = model(inputs)
		losses =  loss_fn(outputs, labels)
		return losses.detach().cpu().tolist() 
	return f

@torch.no_grad()
def get_batch_grad_prod_fn(model, 
						   ref_input, ref_label,
						   loss_fn=nn.CrossEntropyLoss(reduction='none'), 
						   device='cuda'):
	model.eval()
	loss_fn = loss_fn if loss_fn.reduction == 'none' else type(loss_fn)(reduction='none')
	model = model.to(device)

	ref_input = ref_input.to(device).unsqueeze(0)
	ref_label = torch.tensor(ref_label).to(device).unsqueeze(0)

	g_ref = compute_param_grads(model, ref_input, ref_label, loss_fn, device)[0]  # shape: [D]

	def f(batch):
		inputs, labels = batch
		inputs, labels = inputs.to(device), labels.to(device)

		grads = compute_param_grads(model, inputs, labels, loss_fn, device)  # [B, D]

		# 计算与参考样本梯度的内积，shape [B]
		grad_inner_prods = grads @ g_ref
		return grad_inner_prods.detach().cpu().tolist() 
	return f

def get_batch_grad_cos_fn(model, 
						   ref_input, ref_label,
						   loss_fn=nn.CrossEntropyLoss(reduction='none'), 
						   device='cuda'):
	model.eval()
	loss_fn = loss_fn if loss_fn.reduction == 'none' else type(loss_fn)(reduction='none')
	model = model.to(device)

	ref_input = ref_input.to(device).unsqueeze(0)
	ref_label = torch.tensor(ref_label).to(device).unsqueeze(0)

	g_ref = compute_param_grads(model, ref_input, ref_label, loss_fn, device)[0]  # shape: [D]

	def f(batch):
		inputs, labels = batch
		inputs, labels = inputs.to(device), labels.to(device)

		grads = compute_param_grads(model, inputs, labels, loss_fn, device)  # [B, D]

		# 计算与参考样本梯度的内积，shape [B]
		batch_gradcos = F.cosine_similarity(grads, g_ref.unsqueeze(0), dim=1)
		return batch_gradcos.detach().cpu().tolist() 
	
	return f





cifar10_cls = [
	'airplane', 'automobile', 'bird', 'cat', 'deer',     
	'dog', 'frog', 'horse', 'ship', 'truck'
]

@torch.no_grad()
def eval_samples(
	model: nn.Module, inputs: Tensor, labels: Tensor, 
	loss_fn = nn.CrossEntropyLoss(reduction='none'), 
	device='cuda'):

	loss_fn = loss_fn if loss_fn.reduction == 'none' \
					else type(loss_fn)(reduction='none')

	inputs = inputs.to(device)
	labels = labels.to(device)
	model = model.to(device)
	model.eval()

	outputs = model(inputs)
	losses = loss_fn(outputs, labels)
	pred_labels = torch.argmax(outputs, dim=1)

	return losses, pred_labels


def eval_single(model, input, label : int, device='cuda'):
	# 增加 batch 维度
	input_ = input.unsqueeze(0)   # (1, C, H, W)
	label_ = torch.tensor([label], dtype=torch.long)  # (1,)

	# 调用函数
	loss, pred_label = eval_samples(model, input_, label_, device=device)

	return loss.item(), pred_label.item()


def plot_topk_info(model, target_ds, fn, ref_input, ref_label,
			   batch_size=512, device='cuda', print_sims = False):
	sim_samples = find_topk_samples(target_ds, fn, k = 16, batch_size = batch_size)
	
	if print_sims:
		for i in range(16):
			print(sim_samples[i][0])
	
	loss, pred_label = eval_single(model, ref_input, ref_label, device=device)
	
	# 画出原图
	img = inverse_trans_cifar10(ref_input)
	img_show(img)
	print(f'loss: {loss:.4f}' + \
		'    true label: ' + cifar10_cls[ref_label] + \
		'    pred label: ' + cifar10_cls[pred_label])

	# 画出16个sim图
	fig, axes = plt.subplots(4, 4, figsize=(8,8))
	axes = axes.flatten()
	for i in range(16):
		sample_idx = sim_samples[i][1]
		img = inverse_trans_cifar10(target_ds[sample_idx][0])
		img_show(img, ax=axes[i])

	# 打印信息
	sim_sample_indices = [sim_samples[i][1] for i in range(16)]

	imgs, labels = zip(*[target_ds[idx] for idx in sim_sample_indices])
	imgs = torch.stack(imgs)     # shape: (16, C, H, W)
	labels = torch.tensor(labels) # shape: (16,)

	# 批量评估
	losses, pred_labels = eval_samples(model, imgs, labels)

	# 打印结果
	for i in range(16):
		print(f'loss: {losses[i]:.4f}' +
			f'    true: {cifar10_cls[labels[i].item()]}' +
			f'    pred: {cifar10_cls[pred_labels[i].item()]}')

	return sim_sample_indices
