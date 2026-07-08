import numpy as np
from PIL import Image, ImageOps
import torch.utils.data as data
import random
import math


class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        data_dir = 'SYSU_DATA_PATH'
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')
        self.train_color_camid = np.load(data_dir + 'train_rgb_resized_camid.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        self.train_thermal_camid = np.load(data_dir + 'train_ir_resized_camid.npy')

        # BGR to RGB
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1, camid1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]], self.train_color_camid[self.cIndex[index]]
        img2,  target2, camid2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]], self.train_thermal_camid[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2, camid1, camid2

    def __len__(self):
        return len(self.train_color_label)


# ====== Mask-aware data augmentation functions ======

def random_crop_and_flip(img, mask, pad=10, crop_h=288, crop_w=144):
    """Synchronized Pad + RandomCrop + RandomHorizontalFlip for image and mask."""
    # Ensure mask is 2D
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    # Pad
    img = Image.fromarray(img)
    mask = Image.fromarray((mask * 255).astype(np.uint8))

    img = ImageOps.expand(img, border=pad, fill=0)
    mask = ImageOps.expand(mask, border=pad, fill=0)

    # RandomCrop
    w, h = img.size
    th, tw = crop_h, crop_w

    if w == tw and h == th:
        i, j = 0, 0
    else:
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

    img = img.crop((j, i, j + tw, i + th))
    mask = mask.crop((j, i, j + tw, i + th))

    # RandomHorizontalFlip
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    img = np.array(img)
    mask = (np.array(mask) > 127).astype(np.uint8)
    return img, mask


def regional_random_grayscale(img, mask, p=0.5, where='body'):
    """BGS: Body Grayscale — 对行人身体区域进行随机灰度化。
    Image and mask undergo the same operation (mask unchanged for grayscale).
    """
    if random.random() > p:
        return img, mask

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    # Convert to grayscale
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


def regional_random_erasing(img, mask, p=0.5, sl=0.1, sh=0.5, r1=0.375, r2=2.66,
                             mean=[123, 116, 103], where='body'):
    """BRE: Body Random Erasing — 掩码约束的随机擦除（仅作用于人体区域）。
    Image and mask are erased in the SAME spatial region.
    """
    if random.random() > p:
        return img, mask

    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]

    H, W, C = img.shape
    area = H * W

    # Select candidate pixels based on where parameter
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

    for _attempt in range(100):
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
            mask_out[y1:y2, x1:x2] = 0

            return out, mask_out

    return img, mask


# ====== SYSUData with mask support ======

class SYSUData_mask(data.Dataset):
    """SYSU-MM01 training dataset with mask support for FRL+SCG.

    Loads 'pure' images (without empty-pedestrian images) and corresponding
    binary masks from YOLO segmentation. Images and masks are resized from
    384x144 to 288x144 during loading.

    Mask-aware augmentations (Pad, RandomCrop, RandomHorizontalFlip, RandomErasing)
    are applied synchronously to both image and mask.
    """

    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        # Load pure images and binary masks (384x144)
        self.train_color_image = np.load(data_dir + 'train_rgb_resized_img_pure.npy')
        self.train_color_mask = np.load(data_dir + 'train_rgb_resized_img_mask.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        self.train_thermal_image = np.load(data_dir + 'train_ir_resized_img_pure.npy')
        self.train_thermal_mask = np.load(data_dir + 'train_ir_resized_img_mask.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')

        # Camid: try to load from npy, otherwise use zeros
        try:
            self.train_color_camid = np.load(data_dir + 'train_rgb_resized_camid.npy')
        except FileNotFoundError:
            print("Warning: train_rgb_resized_camid.npy not found, using zeros")
            self.train_color_camid = np.zeros(len(self.train_color_label), dtype=np.int64)

        try:
            self.train_thermal_camid = np.load(data_dir + 'train_ir_resized_camid.npy')
        except FileNotFoundError:
            print("Warning: train_ir_resized_camid.npy not found, using zeros")
            self.train_thermal_camid = np.zeros(len(self.train_thermal_label), dtype=np.int64)

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        # ---- RGB ----
        img1 = self.train_color_image[self.cIndex[index]]           # (384, 144, 3)
        mask1 = self.train_color_mask[self.cIndex[index]]           # (1, 384, 144)
        label1 = self.train_color_label[self.cIndex[index]]
        camid1 = self.train_color_camid[self.cIndex[index]]

        # ---- IR ----
        img2 = self.train_thermal_image[self.tIndex[index]]         # (384, 144, 3)
        mask2 = self.train_thermal_mask[self.tIndex[index]]         # (1, 384, 144)
        label2 = self.train_thermal_label[self.tIndex[index]]
        camid2 = self.train_thermal_camid[self.tIndex[index]]

        # ---- Resize from 384x144 to 288x144 ----
        img1 = np.array(Image.fromarray(img1).resize((144, 288), Image.LANCZOS))
        img2 = np.array(Image.fromarray(img2).resize((144, 288), Image.LANCZOS))

        # Squeeze mask channel dim for augmentation
        mask1 = mask1[0] if mask1.ndim == 3 else mask1  # (384, 144) or (288, 144)
        mask2 = mask2[0] if mask2.ndim == 3 else mask2

        # Resize masks
        mask1 = np.array(Image.fromarray((mask1 * 255).astype(np.uint8)).resize((144, 288), Image.LANCZOS))
        mask1 = (mask1 > 127).astype(np.uint8)
        mask2 = np.array(Image.fromarray((mask2 * 255).astype(np.uint8)).resize((144, 288), Image.LANCZOS))
        mask2 = (mask2 > 127).astype(np.uint8)

        # ---- Synchronized augmentations (SFGPN order: Grayscale → Erasing → Crop+Flip) ----
        # BGS: Body Grayscale (p=0.8, body region only)
        img1, mask1 = regional_random_grayscale(img1, mask1, where="body", p=0.8)
        # BRE: Body Random Erasing (p=0.8, body region only, SFGPN params)
        img1, mask1 = regional_random_erasing(img1, mask1, p=0.8, sl=0.1, sh=0.5,
                                               r1=0.375, r2=2.66, mean=[123, 116, 103], where='body')
        # Crop + Flip (pad=10, crop to 288×144)
        img1, mask1 = random_crop_and_flip(img1, mask1, pad=10, crop_h=288, crop_w=144)

        # BGS + BRE + CropFlip for IR
        img2, mask2 = regional_random_grayscale(img2, mask2, where="body", p=0.8)
        img2, mask2 = regional_random_erasing(img2, mask2, p=0.8, sl=0.1, sh=0.5,
                                               r1=0.375, r2=2.66, mean=[123, 116, 103], where='body')
        img2, mask2 = random_crop_and_flip(img2, mask2, pad=10, crop_h=288, crop_w=144)

        # ---- Final transform (ToTensor + Normalize for image only) ----
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        mask1_tensor = torch.from_numpy(mask1).unsqueeze(0).float()
        mask2_tensor = torch.from_numpy(mask2).unsqueeze(0).float()

        return img1, mask1_tensor, label1, camid1, img2, mask2_tensor, label2, camid2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        data_dir='REGDB_DATA_PATH'
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


import torch
