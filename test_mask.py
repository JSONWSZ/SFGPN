import numpy as np
import matplotlib.pyplot as plt
import random

# 路径配置
npy_path = '../Datasets/SYSU-MM01-npy/'

# 读取原图和mask
rgb_img = np.load(npy_path + 'train_rgb_resized_img_pure.npy')
rgb_mask = np.load(npy_path + 'train_rgb_resized_img_mask.npy')

ir_img = np.load(npy_path + 'train_ir_resized_img_pure.npy')
ir_mask = np.load(npy_path + 'train_ir_resized_img_mask.npy')

print("RGB img shape:", rgb_img.shape)   # (N, H, W, 3)
print("RGB mask shape:", rgb_mask.shape) # (N, H, W)
print("IR img shape:", ir_img.shape)     # (N, H, W, 3) 灰度图一般也会存成3通道
print("IR mask shape:", ir_mask.shape)   # (N, H, W)


def visualize_with_overlay(images, masks, title, num_samples=3):
    """随机显示原图、mask和叠加效果"""
    indices = random.sample(range(len(images)), num_samples)
    for idx in indices:
        img = images[idx]
        mask = masks[idx]

        plt.figure(figsize=(12, 4))

        # 原图
        plt.subplot(1, 3, 1)
        plt.imshow(img.astype(np.uint8))
        plt.title(f"{title} Original #{idx}")
        plt.axis('off')

        # 掩码
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.axis('off')

        # 原图 + 掩码叠加
        plt.subplot(1, 3, 3)
        plt.imshow(img.astype(np.uint8))
        plt.imshow(mask, cmap='Reds', alpha=0.4)  # 🔥 红色半透明叠加
        plt.title("Overlay")
        plt.axis('off')

        plt.show()


# 可视化RGB和IR
visualize_with_overlay(rgb_img, rgb_mask, "RGB", num_samples=3)
visualize_with_overlay(ir_img, ir_mask, "IR", num_samples=3)
