from __future__ import print_function
import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

# 引入 Grad-CAM 库
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 引入你的模型定义 (确保 model.py 在同级目录下)
from model import embed_net

# ================= 配置区域 (请在此处修改路径和ID) =================

# 1. 原始数据集根目录 (最后不要带斜杠)
DATASET_ROOT = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01"

# 2. 输出保存目录 (会自动创建)
OUTPUT_ROOT_ORIGIN = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01-Grad-SFGPN1"  # 分支1: Origin
OUTPUT_ROOT_DILATED = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01-Grad-SFGPN2"  # 分支2: Dilated
OUTPUT_ROOT_PPM = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01-Grad-SFGPN3"  # 分支3: PPM
OUTPUT_ROOT_FUSED = r"/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01-Grad-SFGPN4"  # 融合: Average

# 3. 指定要画图的测试集 ID 列表 (手动填写)
TARGET_IDS = ['0006']

# 4. 模型权重路径
MODEL_PATH = r"save_model/sysu_agw_p4_n6_lr_0.1_seed_0_best.t"

# 5. 其他参数
IMG_H = 384
IMG_W = 144
GPU_ID = '3'

# ===================================================================

# 设置 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 图像预处理 (与你的训练一致)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# 注意：Grad-CAM库通常需要 float32 的 tensor，且经过归一化

# --- 核心：模型包装器 ---
# 这是一个关键类。
# 它可以解决你的模型在 Eval 模式下不输出 Logits 的问题，
# 并且解决 1->3 的 Batch 维度变换问题。
# --- 修改后的 Wrapper 类 ---
class ReIDGradCAMWrapper(nn.Module):
    def __init__(self, model):
        super(ReIDGradCAMWrapper, self).__init__()
        self.model = model
        self.current_modal = 1  # 默认为可见光 (1: VIS, 2: IR)

    def set_modal(self, modal_id):
        """在外部调用此方法来切换模态"""
        self.current_modal = modal_id

    def forward(self, x):
        # Batch Trick: 仍然取第一张图
        x_input = x[0:1, :, :, :]

        # === 动态选择模态分支 ===
        if self.current_modal == 2:
            # 这里的输入 x_input 已经是预处理好的张量
            # 你的 thermal_module 负责提取红外浅层特征
            x_feat = self.model.thermal_module(x_input)
        else:
            # visible_module 负责提取可见光浅层特征
            x_feat = self.model.visible_module(x_input)

        # === 接下来的骨干网络部分 (与模态无关，是共享权重的) ===
        x_feat = self.model.base_resnet.base.layer1(x_feat)
        x_feat = self.model.base_resnet.base.layer2(x_feat)
        x_feat = self.model.base_resnet.base.layer3(x_feat)

        # 多尺度模块 (产生 Batch*3 的数据)
        x_feat = self.model.multiscale(x_feat)

        # Layer 4 (目标层)
        x_feat = self.model.base_resnet.base.layer4(x_feat)

        # 后处理 (Pooling + BN + Classifier)
        if self.model.gm_pool == 'on':
            b, c, h, w = x_feat.shape
            x_feat_view = x_feat.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x_feat_view ** p, dim=-1) + 1e-12) ** (1 / p)
        else:
            x_pool = self.model.avgpool(x_feat)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.model.bottleneck(x_pool)
        logits = self.model.classifier(feat)

        return logits

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main():
    print('==> Building model..')
    # 初始化你的模型，注意 n_class 需要和你训练时一致 (例如 395)
    # 这里你需要确认 n_class 是多少，这里暂用你代码里的 395
    n_class = 395
    net = embed_net(n_class, gm_pool='off', arch='resnet50')

    # 加载权重
    print(f'==> Loading checkpoint {MODEL_PATH}')
    if os.path.isfile(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device,weights_only=False)
        net.load_state_dict(checkpoint['net'])
    else:
        print("Error: No checkpoint found!")
        return

    net.to(device)
    net.eval()  # 设为评估模式 (固定 BN)

    # 包装模型，使其适配 Grad-CAM 库
    wrapper_model = ReIDGradCAMWrapper(net).to(device)

    # 指定目标层：ResNet 的 layer4 的最后一个 block
    # 你的模型结构是: net -> base_resnet -> base -> layer4
    target_layers = [net.base_resnet.base.layer4[-1]]

    # 初始化 GradCAM++
    cam = GradCAMPlusPlus(model=wrapper_model, target_layers=target_layers)

    print(f'==> Start processing IDs: {TARGET_IDS}')

    # 遍历所有摄像头目录
    # SYSU 结构: cam1/0001/xxx.jpg
    cam_dirs = [d for d in os.listdir(DATASET_ROOT) if d.startswith('cam')]

    total_images_processed = 0

    for cam_dir in cam_dirs:
        cam_path = os.path.join(DATASET_ROOT, cam_dir)
        if not os.path.isdir(cam_path): continue

        # === 修改点：判断模态 ===
        # SYSU-MM01 标准: cam3, cam6 是红外; 其他是可见光
        if 'cam3' in cam_dir or 'cam6' in cam_dir:
            current_modal = 2  # IR 模式
            # print(f"Switching to IR mode for {cam_dir}")
        else:
            current_modal = 1  # VIS 模式
            # print(f"Switching to VIS mode for {cam_dir}")

        # 这一步非常关键：告诉 Wrapper 接下来进来的图片是什么模态
        wrapper_model.set_modal(current_modal)

        for pid in TARGET_IDS:
            pid_path = os.path.join(cam_path, pid)
            if not os.path.isdir(pid_path):
                continue

            # 遍历该 ID 下的所有图片
            for img_name in os.listdir(pid_path):
                if not (img_name.endswith('.jpg') or img_name.endswith('.bmp')):
                    continue

                img_path = os.path.join(pid_path, img_name)

                # 1. 读取并预处理图片
                rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]  # BGR -> RGB
                rgb_img = cv2.resize(rgb_img, (IMG_W, IMG_H))
                rgb_img_float = np.float32(rgb_img) / 255.0

                # 预处理为 Tensor (1, 3, H, W)
                input_tensor = preprocess_image(rgb_img_float,
                                                mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]).to(device)

                # 2. Batch Trick: 复制 3 份
                # 我们告诉 Grad-CAM 我们有 3 张图。
                # 实际上Wrapper只会算第1张，但在内部通过多尺度模块变成了3张特征图。
                # 这样 Grad-CAM 就能收到 3 张图的梯度，对应 3 个分支。
                input_tensor_batched = input_tensor.repeat(3, 1, 1, 1)

                # 3. 计算 Grad-CAM
                # targets=None 表示取预测分数最高的类作为目标类
                grayscale_cams = cam(input_tensor=input_tensor_batched, targets=None)

                # grayscale_cams 的 shape 是 (3, H, W)
                # Index 0: Origin 分支
                # Index 1: Dilated (空洞) 分支
                # Index 2: PPM (池化) 分支

                cam_origin = grayscale_cams[0, :]
                cam_dilated = grayscale_cams[1, :]
                cam_ppm = grayscale_cams[2, :]

                # 4. 融合 (Average)
                # 1. 先计算平均值
                cam_fused = (cam_origin + cam_dilated + cam_ppm) / 3.0

                # 2. 【关键改进】进行 Min-Max 归一化，将其拉伸回 [0, 1]
                # 减去最小值，除以最大值（加上微小量 1e-7 防止除以 0）
                cam_fused = cam_fused - np.min(cam_fused)
                cam_fused = cam_fused / (np.max(cam_fused) + 1e-7)

                # 5. 可视化叠加
                vis_origin = show_cam_on_image(rgb_img_float, cam_origin, use_rgb=True)
                vis_dilated = show_cam_on_image(rgb_img_float, cam_dilated, use_rgb=True)
                vis_ppm = show_cam_on_image(rgb_img_float, cam_ppm, use_rgb=True)
                vis_fused = show_cam_on_image(rgb_img_float, cam_fused, use_rgb=True)

                # 6. 保存结果
                # 构建相对路径: camX/PID/img.jpg
                rel_dir = os.path.join(cam_dir, pid)

                # 确保保存目录存在
                save_dir_1 = os.path.join(OUTPUT_ROOT_ORIGIN, rel_dir)
                save_dir_2 = os.path.join(OUTPUT_ROOT_DILATED, rel_dir)
                save_dir_3 = os.path.join(OUTPUT_ROOT_PPM, rel_dir)
                save_dir_4 = os.path.join(OUTPUT_ROOT_FUSED, rel_dir)

                ensure_dir(save_dir_1)
                ensure_dir(save_dir_2)
                ensure_dir(save_dir_3)
                ensure_dir(save_dir_4)

                # 保存为 BGR 格式 (OpenCV 默认)
                cv2.imwrite(os.path.join(save_dir_1, img_name), cv2.cvtColor(vis_origin, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_dir_2, img_name), cv2.cvtColor(vis_dilated, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_dir_3, img_name), cv2.cvtColor(vis_ppm, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_dir_4, img_name), cv2.cvtColor(vis_fused, cv2.COLOR_RGB2BGR))

                total_images_processed += 1

    print(f"Finished! Processed {total_images_processed} images.")
    print(
        f"Results saved to:\n1. {OUTPUT_ROOT_ORIGIN}\n2. {OUTPUT_ROOT_DILATED}\n3. {OUTPUT_ROOT_PPM}\n4. {OUTPUT_ROOT_FUSED}")


if __name__ == '__main__':
    main()