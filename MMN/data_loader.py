import math
import random
import numpy as np
import torch
from PIL import Image, ImageOps
import torch.utils.data as data
import os


# ========== 数据增强函数（同步图像+掩码） ==========

def random_crop_and_flip(img, mask, pad=10, crop_h=384, crop_w=192):
    """对图像和 mask 进行同步 Pad、RandomCrop、RandomHorizontalFlip"""
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    img = Image.fromarray(img)
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
    mask_pil = Image.fromarray(mask_uint8)

    img = ImageOps.expand(img, border=pad, fill=0)
    mask_pil = ImageOps.expand(mask_pil, border=pad, fill=0)

    w, h = img.size
    th, tw = crop_h, crop_w

    if w == tw and h == th:
        i, j = 0, 0
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    img = img.crop((j, i, j + tw, i + th))
    mask_pil = mask_pil.crop((j, i, j + tw, i + th))

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

    img = np.array(img)
    mask_out = (np.array(mask_pil) > 127).astype(np.uint8)
    return img, mask_out


def regional_random_grayscale(img, mask, p=0.8, where='body'):
    """BGS (Body Grayscale): 对行人身体区域进行随机灰度化"""
    if random.random() > p:
        return img, mask

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    gray = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    gray_3c = np.stack([gray, gray, gray], axis=-1)

    if where == 'body':
        region = (mask == 1)
        out = img.copy()
        out[region] = gray_3c[region]
    elif where == 'background':
        region = (mask == 0)
        out = img.copy()
        out[region] = gray_3c[region]
    elif where == 'all':
        out = gray_3c.copy()
    else:
        raise ValueError("where must be 'body', 'background', or 'all'")

    return out, mask


def regional_random_erasing(img, mask, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.33,
                            mean=None, where='body'):
    """掩码感知的随机擦除，擦除图像区域的同时更新掩码"""
    if mean is None:
        mean = [123, 116, 103]
    if random.random() > p:
        return img, mask

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    H, W, C = img.shape
    area = H * W

    if where == 'body':
        candidate_pixels = np.argwhere(mask == 1)
    elif where == 'background':
        candidate_pixels = np.argwhere(mask == 0)
    elif where == 'all':
        candidate_pixels = np.argwhere(np.ones_like(mask, dtype=np.uint8))
    else:
        raise ValueError("where must be 'body', 'background', or 'all'")

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
            if where in ['body', 'background']:
                mask_out[y1:y2, x1:x2] = 0

            return out, mask_out

    return img, mask


# ========== 掩码感知的SYSU数据集 ==========

class SYSUData_mask(data.Dataset):
    """加载SFGPN数据路径下的pure图像和掩码，resize到192×384"""
    def __init__(self, data_dir, transform_train=None,
                 colorIndex=None, thermalIndex=None):

        # 加载 pure 图像（清洗后的原始图像）
        self.train_color_pure = np.load(data_dir + 'train_rgb_resized_img_pure.npy')
        self.train_color_mask = np.load(data_dir + 'train_rgb_resized_img_mask.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        self.train_thermal_pure = np.load(data_dir + 'train_ir_resized_img_pure.npy')
        self.train_thermal_mask = np.load(data_dir + 'train_ir_resized_img_mask.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        self.transform_train = transform_train

    def __getitem__(self, index):
        # RGB
        rgb_pure = self.train_color_pure[self.cIndex[index]]
        rgb_mask = self.train_color_mask[self.cIndex[index]]
        vis_label = self.train_color_label[self.cIndex[index]]

        # IR
        ir_pure = self.train_thermal_pure[self.tIndex[index]]
        ir_mask = self.train_thermal_mask[self.tIndex[index]]
        ir_label = self.train_thermal_label[self.tIndex[index]]

        # --- 确保掩码为二值 (H, W) ---
        if rgb_mask.ndim == 3 and rgb_mask.shape[0] == 1:
            rgb_mask = rgb_mask[0]
        if ir_mask.ndim == 3 and ir_mask.shape[0] == 1:
            ir_mask = ir_mask[0]
        # 如果掩码值是0/255，归一化为0/1
        if rgb_mask.max() > 1.0:
            rgb_mask = (rgb_mask > 127).astype(np.uint8)
        if ir_mask.max() > 1.0:
            ir_mask = (ir_mask > 127).astype(np.uint8)

        # --- Resize 图像和掩码: 144→192 (width) ---
        rgb_pure_pil = Image.fromarray(rgb_pure)
        rgb_pure_pil = rgb_pure_pil.resize((192, 384), Image.LANCZOS)
        rgb_pure = np.array(rgb_pure_pil)

        rgb_mask_pil = Image.fromarray((rgb_mask * 255).astype(np.uint8))
        rgb_mask_pil = rgb_mask_pil.resize((192, 384), Image.NEAREST)
        rgb_mask = (np.array(rgb_mask_pil) > 127).astype(np.uint8)

        ir_pure_pil = Image.fromarray(ir_pure)
        ir_pure_pil = ir_pure_pil.resize((192, 384), Image.LANCZOS)
        ir_pure = np.array(ir_pure_pil)

        ir_mask_pil = Image.fromarray((ir_mask * 255).astype(np.uint8))
        ir_mask_pil = ir_mask_pil.resize((192, 384), Image.NEAREST)
        ir_mask = (np.array(ir_mask_pil) > 127).astype(np.uint8)

        # --- RGB增强（同步图像+掩码），顺序与SFGPN一致: BGS → BRE → Crop+Flip ---
        vis_aug, vis_mask = regional_random_grayscale(rgb_pure, rgb_mask, p=0.8, where='body')
        vis_aug, vis_mask = regional_random_erasing(vis_aug, vis_mask, p=0.8,
                                                     sl=0.1, sh=0.5, r1=0.375, r2=2.66,
                                                     mean=[123, 116, 103])
        vis_aug, vis_mask = random_crop_and_flip(vis_aug, vis_mask,
                                                  pad=10, crop_h=384, crop_w=192)
        vis_aug = self.transform_train(vis_aug)
        vis_mask_tensor = torch.from_numpy(vis_mask).unsqueeze(0).float()

        # --- IR增强（同步图像+掩码），顺序与SFGPN一致: BGS → BRE → Crop+Flip ---
        ir_aug, ir_mask = regional_random_grayscale(ir_pure, ir_mask, p=0.8, where='body')
        ir_aug, ir_mask = regional_random_erasing(ir_aug, ir_mask, p=0.8,
                                                   sl=0.1, sh=0.5, r1=0.375, r2=2.66,
                                                   mean=[123, 116, 103])
        ir_aug, ir_mask = random_crop_and_flip(ir_aug, ir_mask,
                                                pad=10, crop_h=384, crop_w=192)
        ir_aug = self.transform_train(ir_aug)
        ir_mask_tensor = torch.from_numpy(ir_mask).unsqueeze(0).float()

        return vis_aug, vis_mask_tensor, vis_label, ir_aug, ir_mask_tensor, ir_label

    def __len__(self):
        return len(self.train_color_label)


# ========== 原始数据加载类（保持兼容） ==========

class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):

        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):

            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((192, 384), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((192, 384), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
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

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (192,384)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)

class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (192,384)):

        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
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
