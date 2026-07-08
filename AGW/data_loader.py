import numpy as np
from PIL import Image, ImageOps
import torch.utils.data as data
import random
import torch


class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        data_dir = '../Datasets/SYSU-MM01/'
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
        data_dir = '../Datasets/RegDB/'
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
   
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.Resampling.LANCZOS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.Resampling.LANCZOS)
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
    def __init__(self, test_img_file, test_label, transform=None, img_size = (144,288)):

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

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size = (144,288)):

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
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label


def sync_pad_crop_flip(img, mask, pad=10, crop_h=288, crop_w=144):
    """
    对图像和 mask 进行同步的 Pad → RandomCrop → RandomHorizontalFlip 操作。
    与 AGW 原始 transform_train 中的数据增强完全一致：
      transforms.Pad(10) + transforms.RandomCrop((288, 144)) + transforms.RandomHorizontalFlip()

    Args:
        img:  numpy array (H, W, 3), uint8
        mask: numpy array (H, W), float32/uint8
        pad:  填充大小，默认 10
        crop_h, crop_w: 裁剪后的高度和宽度，默认 (288, 144)

    Returns:
        img_aug:  PIL Image (RGB)
        mask_aug: numpy array (H, W), uint8 (0 or 1)
    """
    # ---- 统一转为 PIL Image ----
    img_pil = Image.fromarray(img)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8) if mask.max() <= 1.0
                                else mask.astype(np.uint8))

    # ---- 同步 Pad ----
    img_pil = ImageOps.expand(img_pil, border=pad, fill=0)
    mask_pil = ImageOps.expand(mask_pil, border=pad, fill=0)

    # ---- 同步 RandomCrop ----
    w, h = img_pil.size  # PIL 尺寸顺序: (宽, 高)
    if w == crop_w and h == crop_h:
        i, j = 0, 0
    else:
        i = random.randint(0, h - crop_h)
        j = random.randint(0, w - crop_w)

    img_pil = img_pil.crop((j, i, j + crop_w, i + crop_h))
    mask_pil = mask_pil.crop((j, i, j + crop_w, i + crop_h))

    # ---- 同步 RandomHorizontalFlip ----
    if random.random() < 0.5:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
        mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)

    # ---- 转回 numpy ----
    mask_aug = (np.array(mask_pil) > 127).astype(np.uint8)

    return img_pil, mask_aug


class SYSUData_mask(data.Dataset):
    """
    SYSU-MM01 数据集加载器（带掩码）。
    从 SFGPN 数据目录加载图像、标签和掩码 —— 三者来自同一预处理流程，
    保证索引一一对应。
    图像和掩码均为 384×144，加载时 resize 到 288×144 匹配 AGW 输入尺寸。
    """

    def __init__(self, data_dir=None, transform=None,
                 colorIndex=None, thermalIndex=None):

        # 从 SFGPN 数据目录加载所有数据（图像 + 标签 + 掩码），保证索引对齐
        data_dir = '../Datasets/SYSU-MM01-npy/'
        self.train_color_image = np.load(data_dir + 'train_rgb_resized_img_pure.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        self.train_color_mask = np.load(data_dir + 'train_rgb_resized_img_mask.npy')

        self.train_thermal_image = np.load(data_dir + 'train_ir_resized_img_pure.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        self.train_thermal_mask = np.load(data_dir + 'train_ir_resized_img_mask.npy')

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def _resize_img_and_mask(self, img, mask, target_h=288, target_w=144):
        """将图像和掩码从 384×144 resize 到 288×144"""
        # --- 图像 resize ---
        img_pil = Image.fromarray(img)
        img_pil = img_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        img_resized = np.array(img_pil)

        # --- 掩码 resize ---
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1.0 else mask.astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8)
        mask_pil = mask_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
        mask_resized = (np.array(mask_pil) > 127).astype(np.float32)

        return img_resized, mask_resized

    def __getitem__(self, index):

        # --- RGB ---
        img1 = self.train_color_image[self.cIndex[index]]        # (384, 144, 3) uint8
        target1 = self.train_color_label[self.cIndex[index]]
        mask1 = self.train_color_mask[self.cIndex[index]]        # (384, 144) 或 (1, 384, 144)

        # --- IR ---
        img2 = self.train_thermal_image[self.tIndex[index]]      # (384, 144, 3) uint8
        target2 = self.train_thermal_label[self.tIndex[index]]
        mask2 = self.train_thermal_mask[self.tIndex[index]]      # (384, 144) 或 (1, 384, 144)

        # --- Resize: 384×144 → 288×144 ---
        img1, mask1 = self._resize_img_and_mask(img1, mask1, target_h=288, target_w=144)
        img2, mask2 = self._resize_img_and_mask(img2, mask2, target_h=288, target_w=144)

        # --- 同步数据增强 (Pad + RandomCrop + RandomHorizontalFlip) ---
        img1, mask1 = sync_pad_crop_flip(img1, mask1, pad=10, crop_h=288, crop_w=144)
        img2, mask2 = sync_pad_crop_flip(img2, mask2, pad=10, crop_h=288, crop_w=144)

        # --- 图像最终变换 (ToTensor + Normalize) ---
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        # --- 掩码转 tensor ---
        mask1_tensor = torch.from_numpy(mask1).unsqueeze(0).float()
        mask2_tensor = torch.from_numpy(mask2).unsqueeze(0).float()

        return img1, mask1_tensor, target1, img2, mask2_tensor, target2

    def __len__(self):
        return len(self.train_color_label)