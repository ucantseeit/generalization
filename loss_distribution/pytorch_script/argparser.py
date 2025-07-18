import argparse

def get_args_parser(add_help=True):
	parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

	# 总体设置
	parser.add_argument("--model", default="resnet18", type=str, help="model name")
	parser.add_argument(
		"--dataset-name", default="cifar10", type=str,
		choices=["imagenet", "cifar10", "mnist", "tinyimagenet"],
		help="Name of the dataset to train (e.g., 'imagenet', 'cifar10', 'tinyimagenet'). Default: cifar10"
	)
	parser.add_argument(
		"--data-path", default="/datasets01/imagenet_full_size/061417/", type=str, # 默认值保持 ImageNet 示例路径
		help="Root path for the dataset. For ImageNet, it's the parent of 'train' and 'val' folders. "
			"For CIFAR10, it's the path to the directory containing CIFAR-10 data files (e.g., cifar-10-batches-py)."
	)
	parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
	parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
	parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
	parser.add_argument(
		"-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
	)
	parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
	parser.add_argument(
		"-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
	)
	parser.add_argument(
		"--test-only",
		dest="test_only",
		help="Only test the model",
		action="store_true",
	)
	parser.add_argument(
		"--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
	)
	parser.add_argument("--use-tensorboard", action="store_true", help="Enable TensorBoard logging")
	
	# 数据预处理与数据增强
	parser.add_argument(
		"--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
	)
	parser.add_argument(
		"--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256 for imagenet)"
	)
	parser.add_argument(
		"--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224 for imagenet)"
	)
	parser.add_argument(
		"--train-crop-size", default=224, type=int, help="the random crop size used for training (default: 224 for imagenet)"
	)
	parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
	parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
	parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
	parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
	parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")
	parser.add_argument(
		"--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
	)
	parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
	parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")

	# 优化器设置
	parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
	parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
	parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
	parser.add_argument(
		"--wd",
		"--weight-decay",
		default=1e-4,
		type=float,
		metavar="W",
		help="weight decay (default: 1e-4)",
		dest="weight_decay",
	)
	parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
	parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
	parser.add_argument(
		"--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
	)
	parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
	parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
	parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
	parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
	parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")

	# 继续训练 & 加载模型参数
	parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
	parser.add_argument("--not-preserve-lrscheduler", dest="not_preserve_lrscheduler", action="store_true", help="discard original lrscheduler")
	parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
		# model-ema 指数滑动平均模型参数
	parser.add_argument(
		"--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
	)
	parser.add_argument(
		"--model-ema-steps",
		type=int,
		default=32,
		help="the number of iterations that controls how often to update the EMA model (default: 32)",
	)
	parser.add_argument(
		"--model-ema-decay",
		type=float,
		default=0.99998,
		help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
	)


	# 高级优化 / 分布式 & ImageNet 特有参数
	parser.add_argument(
		"--cache-dataset",
		dest="cache_dataset",
		help="Cache the datasets for quicker initialization. It also serializes the transforms",
		action="store_true",
	)
	parser.add_argument(
		"--sync-bn",
		dest="sync_bn",
		help="Use sync batch norm",
		action="store_true",
	)
		# distributed training parameters
	parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
	parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
		# Mixed precision training parameters
	parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

	parser.add_argument(
		"--norm-weight-decay",
		default=None,
		type=float,
		help="weight decay for Normalization layers (default: None, same value as --wd)",
	)
	parser.add_argument(
		"--bias-weight-decay",
		default=None,
		type=float,
		help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
	)
	parser.add_argument(
		"--transformer-embedding-decay",
		default=None,
		type=float,
		help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
	)
	parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
	parser.add_argument(
		"--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
	)
	parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")


	return parser