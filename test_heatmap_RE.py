from __future__ import print_function
import argparse, os, random, math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model import embed_net
from utils import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps

# ----------------------------
# 参数设置
# ----------------------------
parser = argparse.ArgumentParser(description='Robust Heatmap Visualization')
parser.add_argument('--arch', default='resnet50', type=str)
parser.add_argument('--resume', default='bag.t', type=str)
parser.add_argument('--model_path', default='save_model/', type=str)
parser.add_argument('--img_w', default=144, type=int)
parser.add_argument('--img_h', default=384, type=int)
parser.add_argument('--gpu', default='0', type=str)
parser.add_argument('--batch_size', default=128, type=int)

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------------
# 路径设置
# ----------------------------
dataset_root = r"E:\python\projects\Datasets\SYSU-MM01-Pure"
mask_root = r"E:\python\projects\Datasets\SYSU-MM01-Pedestrian"
save_heatmap_root = r"E:\python\projects\Datasets\SYSU-MM01-heatmap-RE"
save_aug_root = r"E:\python\projects\Datasets\SYSU-MM01-384-RE"

for root_dir in [save_heatmap_root, save_aug_root]:
    os.makedirs(root_dir, exist_ok=True)

# ----------------------------
# 模型加载
# ----------------------------
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

net = embed_net(395, gm_pool='off', arch=args.arch)
net.to(device)
cudnn.benchmark = True

model_path = os.path.join(args.model_path, args.resume)
if os.path.isfile(model_path):
    checkpoint = torch.load(model_path, weights_only=False)
    net.load_state_dict(checkpoint['net'])
    print(f"Loaded checkpoint {args.resume} (epoch {checkpoint['epoch']})")
else:
    raise FileNotFoundError(f"No checkpoint found at {model_path}")
net.eval()

# ----------------------------
# 数据增强函数定义
# ----------------------------

def regional_random_erasing(img_np, mask_np,
                            sl=0.1, sh=0.5, r1=0.375, r2=2.66, mean=[123,116,103]):
    """仅作用于人体区域的擦除"""
    H, W, C = img_np.shape
    area = H * W
    body_pixels = np.argwhere(mask_np == 1)
    if body_pixels.shape[0] == 0:
        return img_np
    for _ in range(100):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, r2)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if h < H and w < W:
            cy, cx = body_pixels[random.randint(0, body_pixels.shape[0]-1)]
            x1 = max(0, cx - w // 2)
            y1 = max(0, cy - h // 2)
            x2 = min(W, x1 + w)
            y2 = min(H, y1 + h)
            out = img_np.copy()
            out[y1:y2, x1:x2, :] = mean
            return out
    return img_np
# ----------------------------
# Heatmap 保存函数
# ----------------------------
def save_heatmap(heatmap_tensor, save_path):
    heatmap = heatmap_tensor.squeeze().cpu().numpy()
    heatmap = np.clip(heatmap, 0, 1)
    plt.figure(figsize=(1.44, 3.84), dpi=100)
    plt.axis('off')
    plt.imshow(heatmap, cmap='jet')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# ----------------------------
# 收集所有图像路径
# ----------------------------
def collect_images(base_dir):
    image_list = []
    for cam_folder in os.listdir(base_dir):
        cam_path = os.path.join(base_dir, cam_folder)
        if not os.path.isdir(cam_path): continue
        for pid_folder in os.listdir(cam_path):
            pid_path = os.path.join(cam_path, pid_folder)
            if not os.path.isdir(pid_path): continue
            for img_name in os.listdir(pid_path):
                if img_name.lower().endswith(('.jpg','.png','.jpeg')):
                    image_list.append((cam_folder, pid_folder, img_name))
    return image_list

def process_batch(batch_imgs, batch_meta, modal):
    """
    batch_imgs: List[np.ndarray]  (H,W,3)
    batch_meta: List[(cam, pid, name)]
    modal: 1=VIS, 2=IR
    """

    # --- stack tensor ---
    batch_tensor = torch.stack([
        normalize(transforms.ToTensor()(Image.fromarray(img)))
        for img in batch_imgs
    ]).to(device)

    # --- forward ---
    if modal == 1:
        feat = net.visible_module(batch_tensor)
        feat = net.base_resnet.base.layer1(feat)
        feat = net.base_resnet.base.layer2(feat)
        heatmaps = net.vis_focus(feat)
    else:
        feat = net.thermal_module(batch_tensor)
        feat = net.base_resnet.base.layer1(feat)
        feat = net.base_resnet.base.layer2(feat)
        heatmaps = net.ir_focus(feat)

    heatmaps = torch.sigmoid(heatmaps)

    # --- save ---
    for i, (cam, pid, name) in enumerate(batch_meta):
        save_dir = os.path.join(save_heatmap_root, cam, pid)
        os.makedirs(save_dir, exist_ok=True)
        save_heatmap(heatmaps[i], os.path.join(save_dir, name))


# 收集 SYSU-MM01 下的所有图片
selected = collect_images(dataset_root)
print(f"Processing ALL images: {len(selected)}")

# ----------------------------
# 主流程
# ----------------------------
batch_size = args.batch_size

vis_imgs, vis_meta = [], []
ir_imgs, ir_meta = [], []

with torch.no_grad():
    for cam, pid, name in selected:
        img_path = os.path.join(dataset_root, cam, pid, name)
        mask_path = os.path.join(mask_root, cam, pid, name)

        img = np.array(Image.open(img_path).convert('RGB')
                       .resize((144, 384), Image.BILINEAR))
        mask = np.array(Image.open(mask_path).convert('L')
                        .resize((144, 384), Image.NEAREST))
        mask = (mask > 0).astype(np.uint8)

        # --- augmentation ---
        img = regional_random_erasing(img, mask)

        # --- save augmented image ---
        aug_dir = os.path.join(save_aug_root, cam, pid)
        os.makedirs(aug_dir, exist_ok=True)
        Image.fromarray(img).save(os.path.join(aug_dir, name))

        # --- collect batch ---
        if cam in ['cam3', 'cam6']:  # IR
            ir_imgs.append(img)
            ir_meta.append((cam, pid, name))
            if len(ir_imgs) == batch_size:
                process_batch(ir_imgs, ir_meta, modal=2)
                ir_imgs, ir_meta = [], []
        else:  # VIS
            vis_imgs.append(img)
            vis_meta.append((cam, pid, name))
            if len(vis_imgs) == batch_size:
                process_batch(vis_imgs, vis_meta, modal=1)
                vis_imgs, vis_meta = [], []

    # --- process remaining ---
    if len(vis_imgs) > 0:
        process_batch(vis_imgs, vis_meta, modal=1)
    if len(ir_imgs) > 0:
        process_batch(ir_imgs, ir_meta, modal=2)


print("✅ Heatmap generation (RE) completed successfully!")
