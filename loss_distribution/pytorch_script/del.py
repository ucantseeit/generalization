import os
import re # 用于正则表达式匹配文件名

def delete_non_multiple_of_five_checkpoints(output_dir, start_epoch_inclusive=40, save_multiple=5):
    """
    删除指定目录中，epoch 编号大于等于 start_epoch_inclusive 且不为 save_multiple 倍数的所有 model_XXX.pth 文件。

    Args:
        output_dir (str): 模型检查点文件所在的目录。
        start_epoch_inclusive (int): 从这个 epoch 编号开始检查并删除。
        save_multiple (int): 只保留 save_multiple 的倍数的 epoch 文件。
    """
    print(f"Scanning directory: {output_dir}")
    print(f"Deleting files where epoch >= {start_epoch_inclusive} and epoch is NOT a multiple of {save_multiple}.")

    files_in_dir = os.listdir(output_dir)
    files_to_delete = []

    # 正则表达式匹配 model_XXX.pth 格式
    # group(1) 会捕获 XXX 部分
    pattern = re.compile(r"model_(\d{3})\.pth")

    for filename in files_in_dir:
        match = pattern.match(filename)
        if match:
            epoch_str = match.group(1)
            try:
                epoch_num = int(epoch_str)
                
                # 检查条件：
                # 1. epoch 编号大于等于起始 epoch
                # 2. epoch 编号不是 save_multiple 的倍数
                if epoch_num >= start_epoch_inclusive and epoch_num % save_multiple != 0:
                    full_path = os.path.join(output_dir, filename)
                    files_to_delete.append(full_path)
            except ValueError:
                # 如果文件名中的数字部分不是有效的整数，则跳过
                continue
    
    if not files_to_delete:
        print("No files found matching the deletion criteria.")
        return

    print("\nFiles to be deleted:")
    for f in files_to_delete:
        print(f"  - {f}")

    confirm = input("\nAre you sure you want to delete these files? (yes/no): ").lower()
    if confirm == 'yes':
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"Deleted: {os.path.basename(f)}")
            except OSError as e:
                print(f"Error deleting {f}: {e}")
        print("\nDeletion complete.")
    else:
        print("Deletion cancelled.")

if __name__ == "__main__":
    # 请将这里的 'your_output_directory_path' 替换为你的实际输出目录
    # 例如：'./cifar10_resnet18_output'
    output_directory = '../model_training_results/cifar10_resnet18' # 替换为你的检查点目录

    # 删除从 model_40.pth 开始，所有非 5 的倍数的 epoch 文件
    # 例如，如果 model_40.pth 存在，但 model_41.pth, model_42.pth, model_43.pth, model_44.pth 存在，它们将被删除
    # 而 model_45.pth 会被保留
    delete_non_multiple_of_five_checkpoints(
        output_dir=output_directory,
        start_epoch_inclusive=40, # 从这个 epoch 开始检查
        save_multiple=10 # 保留 5 的倍数
    )