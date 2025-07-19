import datetime
import os
import time
from unittest import TextTestRunner
import warnings

import torch
import torch.utils.data

import utils
import models
from argparser import get_args_parser


from torch import nn
from torch.utils.data.dataloader import default_collate

from transforms import get_mixup_cutmix

from torch.utils.tensorboard import SummaryWriter

from datasets import load_data


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None, writer=None):
	model.train()
	metric_logger = utils.MetricLogger(delimiter="  ")
	metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
	metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

	header = f"Epoch: [{epoch}]"
	losses = []
	for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
		start_time = time.time()
		image, target = image.to(device), target.to(device)
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			output = model(image)
			loss = criterion(output, target)

		optimizer.zero_grad()
		if scaler is not None:
			# scaler逻辑
			scaler.scale(loss).backward()
			if args.clip_grad_norm is not None:
				# we should unscale the gradients of optimizer's assigned params if do gradient clipping
				scaler.unscale_(optimizer)
				nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			if args.clip_grad_norm is not None:
				nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
			optimizer.step()

		# ema逻辑
		if model_ema and i % args.model_ema_steps == 0:
			model_ema.update_parameters(model)
			if epoch < args.lr_warmup_epochs:
				# Reset ema buffer to keep copying weights during warmup period
				model_ema.n_averaged.fill_(0)

		# log逻辑
		acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
		batch_size = image.shape[0]
		current_lr = optimizer.param_groups[0]["lr"]
		metric_logger.update(loss=loss.item(), lr=current_lr)
		metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
		metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
		metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

		# tensorboard逻辑
		if writer:
			global_step = epoch * len(data_loader) + i
			writer.add_scalar('Loss/Train_Step', loss.item(), global_step)
			writer.add_scalar('Acc/Train_Step', acc1.item(), global_step)
			writer.add_scalar('LearningRate/Train_Step', current_lr, global_step)


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix="", writer=None, epoch=0):
	model.eval()
	metric_logger = utils.MetricLogger(delimiter="  ")
	header = f"Test: {log_suffix}"

	num_processed_samples = 0
	with torch.inference_mode():
		for image, target in metric_logger.log_every(data_loader, print_freq, header):
			image = image.to(device, non_blocking=True)
			target = target.to(device, non_blocking=True)
			output = model(image)
			loss = criterion(output, target)

			acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
			# FIXME need to take into account that the datasets
			# could have been padded in distributed setup
			batch_size = image.shape[0]
			metric_logger.update(loss=loss.item())
			metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
			metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
			num_processed_samples += batch_size
	# gather the stats from all processes

	num_processed_samples = utils.reduce_across_processes(num_processed_samples)
	if (
		hasattr(data_loader.dataset, "__len__")
		and len(data_loader.dataset) != num_processed_samples
		and torch.distributed.get_rank() == 0
	):
		# See FIXME above
		warnings.warn(
			f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
			"samples were used for the validation, which might bias the results. "
			"Try adjusting the batch size and / or the world size. "
			"Setting the world size to 1 is always a safe bet."
		)

	metric_logger.synchronize_between_processes()

	print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")

	# 新增: TensorBoard 记录测试指标 (按 epoch 记录)
	if writer:
		writer.add_scalar(f'Loss/Test_Epoch{log_suffix}', metric_logger.loss.global_avg, epoch)
		writer.add_scalar(f'Acc/Test_Epoch{log_suffix}', metric_logger.acc1.global_avg, epoch)
	
	return metric_logger.acc1.global_avg

def main(args):
	if args.output_dir:
		utils.mkdir(args.output_dir)

	utils.init_distributed_mode(args)
	print(args)

	device = torch.device(args.device)

	if args.use_deterministic_algorithms:
		torch.backends.cudnn.benchmark = False
		torch.use_deterministic_algorithms(True)
	else:
		torch.backends.cudnn.benchmark = True

	dataset, dataset_test, train_sampler, test_sampler, num_classes = load_data(args)

	# tensorboard逻辑
	writer = None
	if args.use_tensorboard:
		log_dir = os.path.join(args.output_dir, "runs")
		utils.mkdir(log_dir) # 确保 runs 目录存在
		writer = SummaryWriter(log_dir=log_dir)
		print(f"TensorBoard logs will be saved to: {log_dir}")

	# mixup_cutmix 逻辑
	mixup_cutmix = None
	if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
		if args.dataset_name.lower() == "imagenet": # 仅对 ImageNet 启用 mixup/cutmix
			mixup_cutmix = get_mixup_cutmix(
				mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes
			)

	if mixup_cutmix is not None:
		def collate_fn(batch):
			return mixup_cutmix(*default_collate(batch))
	else:
		collate_fn = default_collate

	# 创建data_loader
	data_loader = torch.utils.data.DataLoader(
		dataset,
		batch_size=args.batch_size,
		sampler=train_sampler,
		num_workers=args.workers,
		pin_memory=True,
		collate_fn=collate_fn,
	)
	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
	)

	# Create model
	# print("Creating model")
	# if args.model == "lenet":
	# 	# Determine in_channels for LeNet based on dataset
	# 	if args.dataset_name.lower() == "mnist":
	# 		in_channels = 1
	# 	else:
	# 		in_channels = 3
	# 	model = LeNet(num_classes=num_classes, in_channels=in_channels)
	# if args.model == 'resnet20':
	# 	model = resnet20()
	# else:
	# 	use_weights = args.weights if args.dataset_name.lower() == "imagenet" else None
	# 	model = torchvision.models.get_model(args.model, weights=use_weights, num_classes=num_classes)
	model = models.get_model(args.dataset_name, args.model, weights=None, num_classes=num_classes)

	model.to(device)

	if args.distributed and args.sync_bn:
		model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

	criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

	custom_keys_weight_decay = []
	if args.bias_weight_decay is not None:
		custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
	if args.transformer_embedding_decay is not None:
		for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
			custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
	parameters = utils.set_weight_decay(
		model,
		args.weight_decay,
		norm_weight_decay=args.norm_weight_decay,
		custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
	)

	# 优化器设置
	opt_name = args.opt.lower()
	if opt_name.startswith("sgd"):
		optimizer = torch.optim.SGD(
			parameters,
			lr=args.lr,
			momentum=args.momentum,
			weight_decay=args.weight_decay,
			nesterov="nesterov" in opt_name,
		)
	elif opt_name == "rmsprop":
		optimizer = torch.optim.RMSprop(
			parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
		)
	elif opt_name == "adamw":
		optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
	else:
		raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

	scaler = torch.cuda.amp.GradScaler() if args.amp else None

	# lr_scheduler设置
	args.lr_scheduler = args.lr_scheduler.lower()
	if args.lr_scheduler == "steplr":
		main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
	elif args.lr_scheduler == "cosineannealinglr":
		main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
		)
	elif args.lr_scheduler == "exponentiallr":
		main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
	else:
		raise RuntimeError(
			f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
			"are supported."
		)

	# lr_warmup设置
	if args.lr_warmup_epochs > 0:
		if args.lr_warmup_method == "linear":
			warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
				optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
			)
		elif args.lr_warmup_method == "constant":
			warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
				optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
			)
		else:
			raise RuntimeError(
				f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
			)
		lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
			optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
		)
	else:
		lr_scheduler = main_lr_scheduler

	# prepare for distributed training 
	# (ddp is abbreviation of DistributedDataParallel )
	model_without_ddp = model
	if args.distributed:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
		model_without_ddp = model.module

	# prepare for model ema (Exponential Moving Average)
	model_ema = None
	if args.model_ema:
		# Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
		# https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
		#
		# total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
		# We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
		# adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
		adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
		alpha = 1.0 - args.model_ema_decay
		alpha = min(1.0, alpha * adjust)
		model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

	# 继续训练
	if args.resume:
		checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
		model_without_ddp.load_state_dict(checkpoint["model"])
		if not args.test_only:
			optimizer.load_state_dict(checkpoint["optimizer"])

			# 如果不保存lrscheduler, 则不加载, 对cosinelr尤其关键
			if not args.not_preserve_lrscheduler:
				lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
			else:
				print("Warning: Learning Rate Scheduler state is not loaded as --not-preserve-lrscheduler was specified.")

		args.start_epoch = checkpoint["epoch"] + 1
		if model_ema:
			model_ema.load_state_dict(checkpoint["model_ema"])
		if scaler:
			scaler.load_state_dict(checkpoint["scaler"])

	# test_only
	if args.test_only:
		# We disable the cudnn benchmarking because it can noticeably affect the accuracy
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True
		if model_ema:
			evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", writer=writer)
		else:
			evaluate(model, criterion, data_loader_test, device=device, writer=writer)

		if writer: # 新增：测试模式下也关闭writer
			writer.close()
		return

	print("Start training")
	start_time = time.time()
	for epoch in range(args.start_epoch, args.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)
		train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, writer=writer)
		lr_scheduler.step()
		evaluate(model, criterion, data_loader_test, device=device, writer=writer, epoch=epoch)
		if model_ema:
			evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA", writer=writer, epoch=epoch)
		if args.output_dir:
			checkpoint = {
				"model": model_without_ddp.state_dict(),
				"optimizer": optimizer.state_dict(),
				"lr_scheduler": lr_scheduler.state_dict(),
				"epoch": epoch,
				"args": args,
			}
			if model_ema:
				checkpoint["model_ema"] = model_ema.state_dict()
			if scaler:
				checkpoint["scaler"] = scaler.state_dict()
			utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
			utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

	total_time = time.time() - start_time
	total_time_str = str(datetime.timedelta(seconds=int(total_time)))
	print(f"Training time {total_time_str}")


if __name__ == "__main__":
	args = get_args_parser().parse_args()
	main(args)
