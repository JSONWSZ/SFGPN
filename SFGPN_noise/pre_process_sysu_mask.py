import numpy as np
from PIL import Image
import os

# 数据集路径配置
ori_path = '../Datasets/SYSU-MM01/'
pure_path = '../Datasets/SYSU-MM01-Pure/'
pedestrian_path = '../Datasets/SYSU-MM01-Pedestrian/'
npy_path = '../Datasets/SYSU-MM01-npy/'

rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

# 加载ID信息（从Pure数据集读取，假设两个数据集ID相同）
file_path_train = os.path.join(ori_path, 'exp/train_id.txt')
file_path_val = os.path.join(ori_path, 'exp/val_id.txt')

with open(file_path_train, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]

with open(file_path_val, 'r') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]

# 合并训练集和验证集
id_train.extend(id_val)


# 为两个数据集收集匹配的文件路径
def collect_matched_files(data_path):
    files_rgb = []
    files_ir = []
    for id in sorted(id_train):
        for cam in rgb_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                # 按文件名排序确保顺序一致
                new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                files_rgb.extend(new_files)

        for cam in ir_cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([os.path.join(img_dir, i) for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    return files_rgb, files_ir


# 收集两个数据集的文件路径（保持严格对应）
pure_rgb, pure_ir = collect_matched_files(pure_path)
pedestrian_rgb, pedestrian_ir = collect_matched_files(pedestrian_path)

# 验证文件对应关系
assert len(pure_rgb) == len(pedestrian_rgb), "RGB图像数量不匹配"
assert len(pure_ir) == len(pedestrian_ir), "IR图像数量不匹配"

# 从Pure数据集生成标签映射（假设两个数据集ID相同）
pid_container = set()
for img_path in pure_ir:
    pid = int(img_path[-13:-9])
    pid_container.add(pid)
pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

# 图像尺寸设置，注意这里重新设置和DEEN一致
fix_image_width = 144
fix_image_height = 384


def process_images(file_list):
    """处理图像并生成对应的标签数组"""
    images = []
    labels = []
    for img_path in file_list:
        img = Image.open(img_path)
        # img.mode → 'RGB'

        # img.size → (W, H)
        img = img.resize((fix_image_width, fix_image_height), Image.Resampling.LANCZOS)
        images.append(np.array(img))

        pid = int(img_path[-13:-9])
        labels.append(pid2label[pid])

        # (H, W, 3)
    return np.array(images), np.array(labels)


def generate_mask(file_list):
    """生成单通道01掩码，shape=[1,H,W]，dtype=float32"""
    masks = []
    for img_path in file_list:
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.Resampling.LANCZOS)
        arr = np.array(img)
        # 大于0的像素点设为1，等于0的为0
        mask = (arr.sum(axis=-1) > 0).astype(np.float32)  # 0/1 float
        mask = np.expand_dims(mask, axis=0)               # shape: (1, H, W)
        masks.append(mask)
    return np.array(masks)  # shape: (N, 1, H, W), dtype=float32



# 处理RGB图像（两个数据集）
pure_rgb_img, rgb_labels = process_images(pure_rgb)
pedestrian_rgb_mask = generate_mask(pedestrian_rgb)  # 🔥 生成01掩码

# 处理IR图像（两个数据集）
pure_ir_img, ir_labels = process_images(pure_ir)
pedestrian_ir_mask = generate_mask(pedestrian_ir)  # 🔥 生成01掩码

# 保存RGB相关文件
np.save(npy_path + 'train_rgb_resized_img_pure.npy', pure_rgb_img)
np.save(npy_path + 'train_rgb_resized_img_mask.npy', pedestrian_rgb_mask)  # 🔥 改名
np.save(npy_path + 'train_rgb_resized_label.npy', rgb_labels)  # 标签文件只需保存一份

# 保存IR相关文件
np.save(npy_path + 'train_ir_resized_img_pure.npy', pure_ir_img)
np.save(npy_path + 'train_ir_resized_img_mask.npy', pedestrian_ir_mask)  # 🔥 改名
np.save(npy_path + 'train_ir_resized_label.npy', ir_labels)  # 标签文件只需保存一份

print("预处理完成！生成以下文件：")
print(f"RGB模态:")
print(f"- {npy_path}train_rgb_resized_img_pure.npy")
print(f"- {npy_path}train_rgb_resized_img_mask.npy")
print(f"- {npy_path}train_rgb_resized_label.npy")
print(f"\nIR模态:")
print(f"- {npy_path}train_ir_resized_img_pure.npy")
print(f"- {npy_path}train_ir_resized_img_mask.npy")
print(f"- {npy_path}train_ir_resized_label.npy")
