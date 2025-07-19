import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pytorch_script.models import get_model


# 数据加载函数
def load_cifar10_data(data_path, batch_size, num_workers):
	"""
	加载 CIFAR-10 数据集并返回训练和测试 DataLoader。
	"""
	print(f"正在加载 CIFAR-10 数据集到 {data_path}...")

	# 数据预处理
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) # CIFAR-10 的均值和标准差
	])

	try:
		train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
		test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)

		train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
		test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

		print("CIFAR-10 数据集加载成功！")
		return train_loader, test_loader
	except Exception as e:
		print(f"加载 CIFAR-10 数据集时发生错误：{e}")
		print("请检查 CIFAR10_DATA_PATH 是否正确，并确保网络连接正常以便下载数据集。")
		return None, None
	

# 模型加载函数
def load_model_state_dict(ds_name, model_name, num_classes, model_path, device):
	# 实例化模型并加载状态字典
	model = get_model(ds_name, model_name, weights=None, num_classes=num_classes)
	
	checkpoint = torch.load(model_path, map_location=device, weights_only=False)

	# 核心修改：检查加载的内容，并提取 model 的 state_dict
	if isinstance(checkpoint, dict) and 'model' in checkpoint:
		# 如果保存的是一个包含 'model' 键的字典
		print(f"  从字典中提取模型状态字典...")
		model_state_dict = checkpoint['model']
	elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
		# 有些框架会把模型的state_dict放在'state_dict'键下
		print(f"  从字典中提取 'state_dict' 键...")
		model_state_dict = checkpoint['state_dict']
	else:
		# 如果直接保存的就是 state_dict
		print(f"  直接加载模型状态字典...")
		model_state_dict = checkpoint
	
	model.load_state_dict(model_state_dict)
	print("提取成功")
	return model



# 模型评估函数
def evaluate_model_performance(model, data_loader, device):
	try:
		model.eval() # 设置模型为评估模式
		model.to(device)

		correct = 0
		total_loss = 0
		total = 0
		criterion = nn.CrossEntropyLoss()
		with torch.no_grad(): # 在评估时不计算梯度
			for images, labels in data_loader:
				images, labels = images.to(device), labels.to(device)
				outputs = model(images)

				loss = criterion(outputs, labels)
				total_loss += loss.item() * images.size(0)

				_, predicted = torch.max(outputs.data, 1)
				correct += (predicted == labels).sum().item()

				total += labels.size(0)
		
		accuracy = 100 * correct / total
		avg_loss = total_loss / total
		return accuracy, avg_loss
	except Exception as e:
		print(f"评估模型 时发生错误：{e}")
		return None

# 模型文件管理函数
def get_sorted_model_paths(model_dir, model_prefix, model_extension):
	"""
	扫描模型目录，获取所有符合命名约定的模型文件路径，并按纪元排序。
	返回一个 (epoch, model_path) 元组的列表。
	"""
	model_files = []
	if not os.path.exists(model_dir):
		print(f"模型目录 {model_dir} 不存在。请确保模型已保存到此目录。")
		os.makedirs(model_dir) # 尝试创建目录，避免后续错误
		return []

	for f_name in os.listdir(model_dir):
		if f_name.startswith(model_prefix) and f_name.endswith(model_extension):
			try:
				# 从文件名中提取 epoch 号码 (例如 'resnet18_cifar10_epoch_X.pt' -> X)
				epoch_str = f_name.replace(model_prefix, '').replace(model_extension, '')
				epoch = int(epoch_str)
				model_files.append((epoch, os.path.join(model_dir, f_name)))
			except ValueError:
				print(f"跳过无法解析的MODEL_PREFIX或MODEL_EXTENSION文件名：{f_name}")
				continue
	
	model_files.sort(key=lambda x: x[0]) # 按 epoch 排序
	return model_files


import seaborn as sns
def plot_acc_losses(epochs, train_accuracies, test_accuracies, train_losses, test_losses):
	fig, axes = plt.subplots(1, 2, figsize=(10, 6))
	axes = axes.flatten()

	sns.lineplot(x=epochs, y=train_accuracies, label='train', ax=axes[0])
	sns.lineplot(x=epochs, y=test_accuracies, label='test', ax=axes[0])
	sns.lineplot(x=epochs, y=train_losses, label='train', ax=axes[1])
	sns.lineplot(x=epochs, y=test_losses, label='test', ax=axes[1])

	axes[0].set_title('Traning Accuraciess & Test Accuracies')
	axes[0].set_xlabel('Epoch')
	axes[0].set_ylabel('Accuracies')
	axes[0].grid(True)

	axes[1].set_title('Traning Losses & Test Losses')
	axes[1].set_xlabel('Epoch')
	axes[1].set_ylabel('Losses')
	axes[1].grid(True)

def  plot_epochs_losses_distribution(ds_type : str, all_model_losses : dict, epochs):
	'''
	# parameters:  
		**ds_type** = 'train' / 'test'  
		**all_model_losses**: a dictionary whose keys are epochs  
						and values are corresponding losses
		**epochs**: a list to specify epochs to plot
	# Examples:	 
		>>> plot_loss_distribution('train', all_model_train_losses, [0, 10, 20, 100, 500])
	'''
	plt.figure(figsize=(15, 15))
	for i in epochs:
		epoch_losses = all_model_losses[i]
		sns.histplot(epoch_losses, kde=False, label=f'Epoch {i}', alpha=0.7)

	plt.title('Distribution of {} Losses Across Different Epochs'\
		   .format('test' if ds_type == 'test' else 'training'))
	plt.xlabel('Loss Value')
	plt.ylabel('Frequency')
	plt.legend()
	plt.grid(True, linestyle='--', alpha=0.6)
	plt.show()

def _calculate_bin_sum_contributions(epoch_losses, bin_edges, total_loss):
    """
    计算每个 bin 内实际 loss 值的总和，并将其按总 loss 进行归一化。

    参数:
        epoch_losses (np.array 或 list): 当前 epoch 的 loss 值列表或数组。
        bin_edges (np.array): 来自 np.histogram 的 bin 边界。
        total_loss (float): 当前 epoch 的所有 loss 值的总和。

    返回:
        np.array: 归一化后的贡献度 (每个 bin 内实际 loss 总和 / 总 loss)。
    """
    # 初始化一个数组，用于存储每个 bin 内 loss 值的累加和
    bin_sum_losses = np.zeros(len(bin_edges) - 1)

    # 遍历每个 loss，将其累加到正确的 bin 中
    for loss in epoch_losses:
        # np.digitize 返回每个值所属 bin 的索引。
        # 结果是 1-based index，对于等于最后一个 bin 边界的值，返回 len(bin_edges)。
        # 所以需要 -1 转换为 0-based index。
        bin_idx = np.digitize(loss, bin_edges) - 1

        # 确保索引在 bin_sum_losses 的有效范围内
        # 例如，如果 loss 恰好等于最后一个 bin 的上边界，np.digitize 可能返回 len(bin_edges)。
        bin_idx = np.clip(bin_idx, 0, len(bin_sum_losses) - 1)

        bin_sum_losses[bin_idx] += loss

    # 将每个 bin 的累加和除以总 loss 进行归一化
    if total_loss == 0:
        return np.zeros_like(bin_sum_losses)
    else:
        return bin_sum_losses / total_loss


# --- 主绘图函数 ---
def plot_losses_distribution_and_contribution(epoch_index, epoch_losses, ax):
    """
    绘制给定 epoch 的 loss 分布 (直方图) 和每个 bin 的实际 loss 贡献度。

    参数:
        epoch_index (int): 要绘制的 epoch 索引。
        num_bins (int): 直方图的 bin 数量。
        ax (matplotlib.axes.Axes): 用于绘图的 Axes 对象。
    """
    # 将 loss 列表转换为 NumPy 数组，方便处理
    epoch_losses = np.array(epoch_losses)
    total_loss = np.sum(epoch_losses) # 使用 np.sum 更安全，处理空数组或全零情况

    # 计算每个 loss bin 中的样本数量，用于分布图
    # hist: 每个 bin 中的样本计数
    # bin_edges: bin 的边界
    hist, bin_edges = np.histogram(epoch_losses)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2 # bin 的中心点，用于条形图定位

    # 绘制分布图 (直方图) 到主 Y 轴 (左侧)
    sns.histplot(epoch_losses, kde=False, color='skyblue', ax=ax, label='Loss Distribution', alpha=0.6)
    ax.set_ylabel('Frequency', color='skyblue')
    ax.tick_params(axis='y', labelcolor='skyblue')

    # 创建第二个 Y 轴，与主轴共享 X 轴 (右侧)
    ax2 = ax.twinx()

    # !!! 调用新的辅助函数计算贡献度 !!!
    contributions = _calculate_bin_sum_contributions(epoch_losses, bin_edges, total_loss)

    # 绘制贡献度 (条形图) 到 ax2
    ax2.bar(bin_centers, contributions, width=(bin_edges[1] - bin_edges[0]) * 0.9, color='lightcoral', edgecolor='black', label='Loss Contribution', alpha=0.7)
    ax2.set_ylabel('Contribution (Loss * Frequency)', color='lightcoral') # 修改标签，更准确
    ax2.tick_params(axis='y', labelcolor='lightcoral')
    # 确保右侧 Y 轴的范围合理，通常贡献度在 0 到 1 之间
    ax2.set_ylim(0, max(contributions) * 1.1 if len(contributions) > 0 and max(contributions) > 0 else 0.1)


    # 设置子图标题和 X 轴标签
    ax.set_title(f'Loss Distribution and Contribution of Epoch {epoch_index} Losses') # 标题显示 epoch 号
    ax.set_xlabel('Loss Value')
    ax.grid(axis='y', linestyle='--', alpha=0.4) # 主轴网格线
    ax2.grid(False) # 禁用第二个 Y 轴的网格线

    # 合并两个轴的图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize='small') # 缩小字体防止拥挤
