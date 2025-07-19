import torch
from torchvision.transforms.functional import InterpolationMode


# def get_module(use_v2):
#     # We need a protected import to avoid the V2 warning in case just V1 is used
#     if use_v2:
#         import torchvision.transforms.v2

#         return torchvision.transforms.v2
#     else:
#         import torchvision.transforms

#         return torchvision.transforms


class ClassificationPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil"
    ):

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(v2.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.append(v2.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True))
        if hflip_prob > 0:
            transforms.append(v2.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(v2.RandAugment(interpolation=interpolation, magnitude=ra_magnitude))
            elif auto_augment_policy == "ta_wide":
                transforms.append(v2.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(v2.AugMix(interpolation=interpolation, severity=augmix_severity))
            else:
                aa_policy = v2.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(v2.AutoAugment(policy=aa_policy, interpolation=interpolation))

        if backend == "pil":
            transforms.append(v2.PILToTensor())

        transforms.extend(
            [
                v2.ToDtype(torch.float, scale=True),
                v2.Normalize(mean=mean, std=std)
            ]
        )
        if random_erase_prob > 0:
            transforms.append(v2.RandomErasing(p=random_erase_prob))

        transforms.append(v2.ToPureTensor())

        self.transforms = v2.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil"
    ):
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(v2.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            v2.Resize(resize_size, interpolation=interpolation, antialias=True),
            v2.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(v2.PILToTensor())

        transforms += [
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize(mean=mean, std=std)
        ]

        transforms.append(v2.ToPureTensor())

        self.transforms = v2.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)

# --- 新增的 CIFAR10 专用预设类 ---
class CIFAR10PresetTrain:
    def __init__(self):
        self.transforms = v2.Compose([
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.PILToTensor(), # CIFAR10 原始数据通常是 PIL Image
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)), # CIFAR10 均值和标准差
            v2.ToPureTensor(), # 确保 v2 输出纯 Tensor
        ])

    def __call__(self, img):
        return self.transforms(img)

class CIFAR10PresetEval:
    def __init__(self):
        self.transforms = v2.Compose([
            v2.PILToTensor(),
            v2.ToDtype(torch.float, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)), # CIFAR10 均值和标准差
            v2.ToPureTensor()
        ])

    def __call__(self, img):
        return self.transforms(img)


# --- MNIST Specific Preset Classes ---
class MNISTPresetTrain:
    def __init__(self):
        self.transforms = v2.Compose([
            v2.ToTensor(), # MNIST is grayscale, so ToTensor will convert to 1 channel
            v2.Pad(2), # Pad MNIST 28x28 to 32x32 for LeNet compatibility
            v2.Normalize((0.1307,), (0.3081,)), # MNIST mean and std for 1 channel
            v2.ToPureTensor(), # Ensure v2 outputs pure Tensor
        ])

    def __call__(self, img):
        return self.transforms(img)

class MNISTPresetEval:
    def __init__(self):
        self.transforms = v2.Compose([
            v2.ToTensor(),
            v2.Pad(2), # Pad MNIST 28x28 to 32x32 for LeNet compatibility
            v2.Normalize((0.1307,), (0.3081,)), # MNIST mean and std for 1 channel
            v2.ToPureTensor(),
        ])

    def __call__(self, img):
        return self.transforms(img)

from torchvision.transforms import v2
class TinyImageNetPresetTrain:
    def __init__(self, *, crop_size=64, resize_size=64, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = v2.Compose([
            v2.RandomResizedCrop(crop_size),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])

    def __call__(self, img):
        return self.transforms(img)

class TinyImageNetPresetEval:
    def __init__(self, *, crop_size=64, resize_size=64, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = v2.Compose([
            v2.Resize(resize_size),
            v2.CenterCrop(crop_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=list(mean), std=list(std)),
        ])

    def __call__(self, img):
        return self.transforms(img)
