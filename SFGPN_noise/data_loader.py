import math
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
import torch.utils.data as data
import os


def add_mask_noise(mask, kernel_size):
    """
    对二值掩码随机施加腐蚀或膨胀（各50%概率）。
    Args:
        mask: np.ndarray, shape (H, W), 值 ∈ {0, 1}
        kernel_size: int, 腐蚀/膨胀的核大小。0 或负数表示不加噪。
    Returns:
        np.ndarray, 加噪后的掩码
    """
    if kernel_size <= 0:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if np.random.rand() < 0.5:
        return cv2.erode(mask, kernel, iterations=1)
    else:
        return cv2.dilate(mask, kernel, iterations=1)


def regional_random_grayscale(img, mask, p=0.5, where='body'):
    """区域灰度化：支持全图、背景、人体区域"""
    if random.random() > p:
        return img, mask

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    # ---- 转为灰度图 ----
    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    gray_3c = np.stack([gray, gray, gray], axis=-1)

    # ---- 根据区域选择灰度化 ----
    if where == 'background':
        region = (mask == 0)
        out = img.copy()
        out[region] = gray_3c[region]

    elif where == 'body':
        region = (mask == 1)
        out = img.copy()
        out[region] = gray_3c[region]

    elif where == 'all':
        # 对整张图像灰度化
        out = gray_3c.copy()

    else:
        raise ValueError("where 必须是 'background'、'body' 或 'all'")

    return out, mask


def regional_random_erasing(img, mask, p=0.5, sl=0.1, sh=0.5, r1=0.375, r2=2.66, mean=[123, 116, 103], where='body'):
    """掩码约束的随机擦除（支持全图、背景、人体区域）"""
    if random.random() > p:
        return img, mask

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    H, W, C = img.shape
    area = H * W

    # ---- 根据模式选择候选像素 ----
    if where == 'body':
        candidate_pixels = np.argwhere(mask == 1)
    elif where == 'background':
        candidate_pixels = np.argwhere(mask == 0)
    elif where == 'all':
        # 对整张图像随机擦除，不依赖mask
        candidate_pixels = np.argwhere(np.ones_like(mask, dtype=np.uint8))
    else:
        raise ValueError("where 必须是 'body'、'background' 或 'all'")

    # 如果该区域没有有效像素，则返回原图
    if candidate_pixels.shape[0] == 0:
        return img, mask

    for attempt in range(100):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, r2)

        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))

        if h < H and w < W:
            cy, cx = candidate_pixels[random.randint(0, candidate_pixels.shape[0] - 1)]

            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)

            out = img.copy()
            out[y1:y2, x1:x2, 0] = mean[0]
            out[y1:y2, x1:x2, 1] = mean[1]
            out[y1:y2, x1:x2, 2] = mean[2]

            mask_out = mask.copy()
            # 只有 body / background 模式才需要更新 mask
            if where in ['body', 'background']:
                mask_out[y1:y2, x1:x2] = 0

            return out, mask_out

    return img, mask


def random_crop_and_flip(img, mask, pad=10):
    # ---- 保证 mask 为二维 ----
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    """
    对图像和 mask 进行同步 Pad、RandomCrop、RandomHorizontalFlip，
    与 torchvision.transforms 实现方式完全一致。
    """
    # ---- Pad ----
    img = Image.fromarray(img)
    mask = Image.fromarray((mask * 255).astype(np.uint8))  # 转为PIL方便操作

    img = ImageOps.expand(img, border=pad, fill=0)
    mask = ImageOps.expand(mask, border=pad, fill=0)

    # ---- RandomCrop ----
    w, h = img.size
    th, tw = 384, 144

    if w == tw and h == th:
        i, j = 0, 0
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    img = img.crop((j, i, j + tw, i + th))
    mask = mask.crop((j, i, j + tw, i + th))

    # ---- RandomHorizontalFlip ----
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # ---- 转回 numpy ----
    img = np.array(img)
    mask = (np.array(mask) > 127).astype(np.uint8)
    return img, mask


class SYSUData_mask(data.Dataset):
    def __init__(self, data_dir,  transform_train=None, where="body",
                 colorIndex=None, thermalIndex=None, noise_kernel=0):
        """
        Args:
            data_dir: 包含 .npy 文件的目录路径
            transform_train: 用于最终张量化和归一化的 transform
            where: str, 'background' 或 'body'，选择灰度化的区域
            colorIndex, thermalIndex: RGB 和 IR 图像采样索引
            noise_kernel: int, 掩码噪声核大小 (0=不加噪, 5=L1, 11=L2, 21=L3)
        """
        # 加载 Pure 图像和 Mask
        self.train_color_pure = np.load(data_dir + 'train_rgb_resized_img_pure.npy')
        self.train_color_mask = np.load(data_dir + 'train_rgb_resized_img_mask.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        self.train_thermal_pure = np.load(data_dir + 'train_ir_resized_img_pure.npy')
        self.train_thermal_mask = np.load(data_dir + 'train_ir_resized_img_mask.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        self.where = where
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.transform_train = transform_train
        self.noise_kernel = noise_kernel

    def __getitem__(self, index):
        # RGB
        rgb_pure = self.train_color_pure[self.cIndex[index]]
        rgb_mask = self.train_color_mask[self.cIndex[index]]
        vis_label = self.train_color_label[self.cIndex[index]]

        # IR
        ir_pure = self.train_thermal_pure[self.tIndex[index]]
        ir_mask = self.train_thermal_mask[self.tIndex[index]]
        ir_label = self.train_thermal_label[self.tIndex[index]]

        # --- RGB增强 ---
        vis_aug, vis_mask = regional_random_grayscale(rgb_pure, rgb_mask, where="body", p=0.8)
        vis_aug, vis_mask = regional_random_erasing(vis_aug, vis_mask, p=0.8)
        vis_aug, vis_mask = random_crop_and_flip(vis_aug, vis_mask)
        # 掩码噪声注入（在crop/flip之后、转tensor之前）
        vis_mask = add_mask_noise(vis_mask, self.noise_kernel)
        vis_aug = self.transform_train(vis_aug)
        vis_mask_tensor = torch.from_numpy(vis_mask).unsqueeze(0).float()

        # --- IR增强 ---
        ir_aug, ir_mask = regional_random_grayscale(ir_pure, ir_mask, where="body", p=0.8)
        ir_aug, ir_mask = regional_random_erasing(ir_aug, ir_mask, p=0.8)
        ir_aug, ir_mask = random_crop_and_flip(ir_aug, ir_mask)
        # 掩码噪声注入
        ir_mask = add_mask_noise(ir_mask, self.noise_kernel)
        ir_aug = self.transform_train(ir_aug)
        ir_mask_tensor = torch.from_numpy(ir_mask).unsqueeze(0).float()

        return vis_aug, vis_mask_tensor, vis_label, ir_aug, ir_mask_tensor, ir_label

    def __len__(self):
        return len(self.train_color_label)


class RegDBData_mask(data.Dataset):
    def __init__(self, data_dir, Pedestrian_dir, trial,
                 transform_train=None,
                 colorIndex=None, thermalIndex=None, noise_kernel=0):

        self.data_dir = data_dir
        self.Pedestrian_dir = Pedestrian_dir

        # === 读取训练集文件 ===
        train_color_list = os.path.join(data_dir, f'idx/train_visible_{trial}.txt')
        train_thermal_list = os.path.join(data_dir, f'idx/train_thermal_{trial}.txt')

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        # === 加载可见光与红外图像 ===
        train_color_image, train_color_mask = [], []
        for path in color_img_file:
            # 原始图像
            img_path = os.path.join(data_dir, path)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((144, 384), Image.Resampling.LANCZOS)
            img_np = np.array(img)
            train_color_image.append(img_np)

            # 掩码生成（根据 Pedestrian_dir）
            ped_path = os.path.join(Pedestrian_dir, path)
            ped_img = Image.open(ped_path).convert('RGB')
            ped_img = ped_img.resize((144, 384), Image.Resampling.LANCZOS)
            ped_np = np.array(ped_img)

            # 若像素值非零则视为行人
            mask = (ped_np.sum(axis=-1) > 0).astype(np.uint8)
            train_color_mask.append(mask)

        train_thermal_image, train_thermal_mask = [], []
        for path in thermal_img_file:
            # 原始红外图像
            img_path = os.path.join(data_dir, path)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((144, 384), Image.Resampling.LANCZOS)
            img_np = np.array(img)
            train_thermal_image.append(img_np)

            # 掩码
            ped_path = os.path.join(Pedestrian_dir, path)
            ped_img = Image.open(ped_path).convert('RGB')
            ped_img = ped_img.resize((144, 384), Image.Resampling.LANCZOS)
            ped_np = np.array(ped_img)

            mask = (ped_np.sum(axis=-1) > 0).astype(np.uint8)
            train_thermal_mask.append(mask)

        # 转为 numpy 数组
        self.train_color_pure = np.array(train_color_image)
        self.train_color_mask = np.array(train_color_mask)
        self.train_color_label = np.array(train_color_label)

        self.train_thermal_pure = np.array(train_thermal_image)
        self.train_thermal_mask = np.array(train_thermal_mask)
        self.train_thermal_label = np.array(train_thermal_label)

        self.transform_train = transform_train
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.noise_kernel = noise_kernel

    # ------------------- 数据增强部分 -------------------

    def __getitem__(self, index):
        # RGB
        rgb_pure = self.train_color_pure[self.cIndex[index]]
        rgb_mask = self.train_color_mask[self.cIndex[index]]
        vis_label = self.train_color_label[self.cIndex[index]]

        # IR
        ir_pure = self.train_thermal_pure[self.tIndex[index]]
        ir_mask = self.train_thermal_mask[self.tIndex[index]]
        ir_label = self.train_thermal_label[self.tIndex[index]]

        # === 可见光增强 ===
        vis_aug, vis_mask = regional_random_grayscale(rgb_pure, rgb_mask, where="body", p=0.8)
        vis_aug, vis_mask = regional_random_erasing(vis_aug, rgb_mask, p=0.8, sl=0.02, sh=0.4)
        vis_aug, vis_mask = random_crop_and_flip(vis_aug, vis_mask)
        # 掩码噪声注入
        vis_mask = add_mask_noise(vis_mask, self.noise_kernel)
        vis_aug = self.transform_train(vis_aug)
        vis_mask_tensor = torch.from_numpy(vis_mask).unsqueeze(0).float()

        # === 红外增强 ===
        ir_aug, ir_mask = regional_random_grayscale(ir_pure, ir_mask, where="body", p=0.8)
        ir_aug, ir_mask = regional_random_erasing(ir_aug, ir_mask, p=0.8, sl=0.02, sh=0.4)
        ir_aug, ir_mask = random_crop_and_flip(ir_aug, ir_mask)
        # 掩码噪声注入
        ir_mask = add_mask_noise(ir_mask, self.noise_kernel)
        ir_aug = self.transform_train(ir_aug)
        ir_mask_tensor = torch.from_numpy(ir_mask).unsqueeze(0).float()

        return (vis_aug, vis_mask_tensor, vis_label,
                ir_aug, ir_mask_tensor, ir_label)

    def __len__(self):
        return len(self.train_color_label)


class LLCMData_mask(data.Dataset):
    def __init__(self, data_dir, Pedestrian_dir,
                 transform_train=None,
                 colorIndex=None, thermalIndex=None):

        self.data_dir = data_dir
        self.Pedestrian_dir = Pedestrian_dir

        # === 读取训练集文件 ===
        train_color_list = os.path.join(data_dir, 'idx/train_vis.txt')
        train_thermal_list = os.path.join(data_dir, 'idx/train_nir.txt')

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        # === 加载可见光与红外图像及掩码 ===
        train_color_image, train_color_mask = [], []
        for path in color_img_file:
            # 原图
            img_path = os.path.join(data_dir, path)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((144, 384), Image.Resampling.LANCZOS)
            img_np = np.array(img)
            train_color_image.append(img_np)

            # 掩码生成
            ped_path = os.path.join(Pedestrian_dir, path)
            ped_img = Image.open(ped_path).convert('RGB')
            ped_img = ped_img.resize((144, 384), Image.Resampling.LANCZOS)
            ped_np = np.array(ped_img)

            mask = (ped_np.sum(axis=-1) > 0).astype(np.uint8)
            # mask = np.expand_dims(mask, axis=0)
            train_color_mask.append(mask)

        train_thermal_image, train_thermal_mask = [], []
        for path in thermal_img_file:
            # 原图
            img_path = os.path.join(data_dir, path)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((144, 384), Image.Resampling.LANCZOS)
            img_np = np.array(img)
            train_thermal_image.append(img_np)

            # 掩码生成
            ped_path = os.path.join(Pedestrian_dir, path)
            ped_img = Image.open(ped_path).convert('RGB')
            ped_img = ped_img.resize((144, 384), Image.Resampling.LANCZOS)
            ped_np = np.array(ped_img)

            mask = (ped_np.sum(axis=-1) > 0).astype(np.uint8)
            # mask = np.expand_dims(mask, axis=0)
            train_thermal_mask.append(mask)

        # 转为 numpy 数组
        self.train_color_pure = np.array(train_color_image)
        self.train_color_mask = np.array(train_color_mask)
        self.train_color_label = np.array(train_color_label)

        self.train_thermal_pure = np.array(train_thermal_image)
        self.train_thermal_mask = np.array(train_thermal_mask)
        self.train_thermal_label = np.array(train_thermal_label)

        self.transform_train = transform_train
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        # RGB
        rgb_pure = self.train_color_pure[self.cIndex[index]]
        rgb_mask = self.train_color_mask[self.cIndex[index]]
        vis_label = self.train_color_label[self.cIndex[index]]

        # IR/NIR
        ir_pure = self.train_thermal_pure[self.tIndex[index]]
        ir_mask = self.train_thermal_mask[self.tIndex[index]]
        ir_label = self.train_thermal_label[self.tIndex[index]]

        # --- RGB增强 ---
        vis_aug, vis_mask = regional_random_grayscale(rgb_pure, rgb_mask, where="body", p=0.8)
        vis_aug, vis_mask = regional_random_erasing(vis_aug, vis_mask, p=0.8)
        vis_aug, vis_mask = random_crop_and_flip(vis_aug, vis_mask)
        vis_aug = self.transform_train(vis_aug)
        vis_mask_tensor = torch.from_numpy(vis_mask).unsqueeze(0).float()

        # --- IR增强 ---
        ir_aug, ir_mask = regional_random_grayscale(ir_pure, ir_mask, where="body", p=0.8)
        ir_aug, ir_mask = regional_random_erasing(ir_aug, ir_mask, p=0.8)
        ir_aug, ir_mask = random_crop_and_flip(ir_aug, ir_mask)
        ir_aug = self.transform_train(ir_aug)
        ir_mask_tensor = torch.from_numpy(ir_mask).unsqueeze(0).float()

        return vis_aug, vis_mask_tensor, vis_label, ir_aug, ir_mask_tensor, ir_label

    def __len__(self):
        return len(self.train_color_label)








class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            # print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class LLCMData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None):
        # Load training images (path) and labels
        train_color_list = data_dir + 'idx/train_vis.txt'
        train_thermal_list = data_dir + 'idx/train_nir.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
            # print(pix_array.shape)
        train_thermal_image = np.array(train_thermal_image)

        # BGR to RGB
        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        # BGR to RGB
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
        self.test_img_file = test_img_file

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label
