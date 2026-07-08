"""
R1-9: FRL 热图定量评估脚本
============================
在 SYSU-MM01 测试集的所有图像上，评估 FRL (FocusModule) 生成的热图
与 YOLO 行人分割掩码之间的空间对齐程度。

5 个指标:
  (1) mIoU   — Mean Intersection over Union（基于 Otsu 二值化热图）
  (2) Dice   — Dice 分数（基于 Otsu 二值化热图）
  (3) Recall — 前景召回率（基于 Otsu 二值化热图）
  (4) BSR    — 背景抑制比（基于连续值热图）
  (5) ACR    — 注意力集中比（基于连续值热图）

输出:
  - 热图: SYSU-MM01-heatmap/camX/PPPP/NNNN.npy（连续值, float32, 384×144）
  - Otsu: SYSU-MM01-heatmap/camX/PPPP/NNNN-otsu.txt（单行浮点数）
  - 汇总: eval_results.txt（所有指标的平均值）

用法:
  python eval_frl_heatmap.py \
      --dataset_path ../Datasets/SYSU-MM01 \
      --pedestrian_path ../Datasets/SYSU-MM01-Pedestrian \
      --heatmap_save_path ../Datasets/SYSU-MM01-heatmap \
      --checkpoint /path/to/model_best.t \
      --gpu 0
"""

from __future__ import print_function
import argparse
import os
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from PIL import Image

from model import embed_net


# ============================================================================
# 1. 图像收集
# ============================================================================

def collect_test_images(data_path):
    """
    从 SYSU-MM01 的 exp/test_id.txt 读取测试 ID，
    收集所有 6 个摄像头下这些 ID 的全部图像。

    Returns:
        list of (cam, pid, img_name)
        例如 [('cam1', '0001', '0001.jpg'), ...]
    """
    file_path = os.path.join(data_path, 'exp', 'test_id.txt')
    with open(file_path, 'r') as f:
        ids = f.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    all_images = []
    # 全部 6 个摄像头: cam1,2,4,5 为 VIS; cam3,6 为 IR
    cameras = ['cam1', 'cam2', 'cam3', 'cam4', 'cam5', 'cam6']

    for pid in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, pid)
            if os.path.isdir(img_dir):
                for img_name in sorted(os.listdir(img_dir)):
                    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                        all_images.append((cam, pid, img_name))

    return all_images


# ============================================================================
# 2. 掩码加载与二值化
# ============================================================================

def load_mask(pedestrian_path, cam, pid, img_name, target_h=384, target_w=144):
    """
    加载行人分割掩码并二值化。

    SYSU-MM01-Pedestrian 中的图像是实例分割结果（非二值图），
    任何像素值之和 > 0 即视为行人区域（参考 data_loader.py 的处理）。

    Args:
        pedestrian_path: SYSU-MM01-Pedestrian 根目录
        cam: 摄像头文件夹名 (如 'cam1')
        pid: 行人 ID 文件夹名 (如 '0001')
        img_name: 图像文件名 (如 '0001.jpg')
        target_h, target_w: 目标尺寸 (H, W)

    Returns:
        mask: np.ndarray, shape (target_h, target_w), dtype=uint8, 值为 0 或 1
        valid: bool, 是否包含有效行人区域
    """
    mask_path = os.path.join(pedestrian_path, cam, pid, img_name)
    if not os.path.exists(mask_path):
        return None, False

    mask_img = Image.open(mask_path).convert('RGB')
    mask_img = mask_img.resize((target_w, target_h), Image.LANCZOS)
    mask_np = np.array(mask_img)

    # 任何非零像素 → 行人
    mask = (mask_np.sum(axis=-1) > 0).astype(np.uint8)

    # 判断是否有有效行人区域
    valid = mask.sum() > 0
    return mask, valid


# ============================================================================
# 3. Otsu 二值化
# ============================================================================

def otsu_binarize(heatmap):
    """
    对连续值热图进行 Otsu 自适应二值化。

    Args:
        heatmap: np.ndarray, shape (H, W), 值域 [0, 1]

    Returns:
        binary_map: np.ndarray, shape (H, W), 值为 0 或 1
        otsu_thresh: float, Otsu 阈值 (映射回 [0, 1])
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    otsu_thresh_255, binary_map = cv2.threshold(
        heatmap_uint8, 0, 1,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_thresh = otsu_thresh_255 / 255.0
    return binary_map.astype(np.uint8), otsu_thresh


# ============================================================================
# 4. 混淆矩阵计算
# ============================================================================

def compute_tp_fp_tn_fn(binary_map, mask):
    """
    基于二值化热图和掩码计算混淆矩阵元素。

    Args:
        binary_map: np.ndarray, 值为 0 或 1  (经 Otsu 二值化的热图)
        mask:       np.ndarray, 值为 0 或 1  (YOLO 分割掩码真值)

    Returns:
        tp, fp, tn, fn: int
    """
    tp = int(np.sum((binary_map == 1) & (mask == 1)))
    fp = int(np.sum((binary_map == 1) & (mask == 0)))
    tn = int(np.sum((binary_map == 0) & (mask == 0)))
    fn = int(np.sum((binary_map == 0) & (mask == 1)))
    return tp, fp, tn, fn


# ============================================================================
# 5. 五项指标计算（单张图像）
# ============================================================================

def compute_metrics_single(heatmap_cont, mask):
    """
    计算单张图像的 5 项热图质量指标。

    Args:
        heatmap_cont: np.ndarray, shape (H, W), 连续值热图 [0, 1]
        mask:         np.ndarray, shape (H, W), 二值掩码 {0, 1}

    Returns:
        dict with keys: iou, dice, recall, bsr, acr, otsu_thresh
    """
    # Otsu 二值化
    binary_map, otsu_thresh = otsu_binarize(heatmap_cont)

    # 混淆矩阵
    tp, fp, _, fn = compute_tp_fp_tn_fn(binary_map, mask)

    # --- (1) IoU ---
    denom_iou = tp + fp + fn
    iou = tp / denom_iou if denom_iou > 0 else 0.0

    # --- (2) Dice ---
    denom_dice = 2 * tp + fp + fn
    dice = (2 * tp) / denom_dice if denom_dice > 0 else 0.0

    # --- (3) 前景召回率 ---
    denom_recall = tp + fn
    recall = tp / denom_recall if denom_recall > 0 else 0.0

    # --- (4) 背景抑制比 (BSR) ---
    # 使用连续值热图计算，上界裁剪到 1000 以防止 inf
    fg_mean = heatmap_cont[mask == 1].mean() if (mask == 1).sum() > 0 else 0.0
    bg_mean = heatmap_cont[mask == 0].mean() if (mask == 0).sum() > 0 else 0.0
    bsr = fg_mean / (bg_mean + 1e-8) if bg_mean > 0 else 1000.0
    bsr = min(bsr, 1000.0)  # 上界裁剪

    # --- (5) 注意力集中比 (ACR) ---
    # 使用连续值热图计算
    all_mean = heatmap_cont.mean()
    acr = fg_mean / (all_mean + 1e-8) if all_mean > 0 else 0.0

    return {
        'iou': iou,
        'dice': dice,
        'recall': recall,
        'bsr': bsr,
        'acr': acr,
        'otsu_thresh': otsu_thresh,
    }


# ============================================================================
# 6. 热图生成（单张图像）
# ============================================================================

def generate_heatmap(net, img_tensor, modal):
    """
    对单张图像生成 FRL 热图。

    Args:
        net: embed_net 模型
        img_tensor: torch.Tensor, shape (1, 3, 384, 144), 已归一化
        modal: 1=VIS, 2=IR

    Returns:
        heatmap: np.ndarray, shape (384, 144), 值域 [0, 1]
    """
    with torch.no_grad():
        if modal == 1:  # VIS
            feat = net.visible_module(img_tensor)
            feat = net.base_resnet.base.layer1(feat)
            feat = net.base_resnet.base.layer2(feat)
            heatmap = net.vis_focus(feat)
        else:  # modal == 2, IR
            feat = net.thermal_module(img_tensor)
            feat = net.base_resnet.base.layer1(feat)
            feat = net.base_resnet.base.layer2(feat)
            heatmap = net.ir_focus(feat)

    # FocusModule 已包含 Sigmoid，输出值域 [0, 1]
    heatmap_np = heatmap.squeeze().cpu().numpy()  # (384, 144)
    heatmap_np = np.clip(heatmap_np, 0.0, 1.0)
    return heatmap_np


# ============================================================================
# 7. 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='R1-9: FRL Heatmap Quantitative Evaluation on SYSU-MM01'
    )
    # 路径参数
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='SYSU-MM01 数据集根目录')
    parser.add_argument('--pedestrian_path', type=str, required=True,
                        help='SYSU-MM01-Pedestrian 行人分割数据集根目录')
    parser.add_argument('--heatmap_save_path', type=str, required=True,
                        help='热图保存目录（将保持与 SYSU-MM01 相同的子目录结构）')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='预训练模型权重路径 (.t 文件)')
    # 模型参数
    parser.add_argument('--arch', default='resnet50', type=str,
                        help='网络架构: resnet18 或 resnet50')
    parser.add_argument('--class_num', default=395, type=int,
                        help='分类数（SYSU: 395）')
    parser.add_argument('--img_w', default=144, type=int, help='图像宽度')
    parser.add_argument('--img_h', default=384, type=int, help='图像高度')
    # 运行参数
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU 设备 ID')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='批处理大小（仅影响进度打印，不影响计算）')
    parser.add_argument('--results_file', default='eval_results.txt', type=str,
                        help='汇总结果输出文件名')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ======================================================================
    # 7a. 加载模型
    # ======================================================================
    print('==> Building model..')
    net = embed_net(args.class_num, gm_pool='off', arch=args.arch, dataset='sysu')
    net.to(device)
    cudnn.benchmark = True

    if os.path.isfile(args.checkpoint):
        print(f'==> Loading checkpoint: {args.checkpoint}')
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['net'])
        print(f'==> Loaded checkpoint (epoch {checkpoint.get("epoch", "?")})')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {args.checkpoint}')
    net.eval()

    # ======================================================================
    # 7b. 图像预处理变换
    # ======================================================================
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    def preprocess_image(img_path):
        """加载图像 → resize → tensor → normalize"""
        img = Image.open(img_path).convert('RGB')
        img = img.resize((args.img_w, args.img_h), Image.BILINEAR)
        img_np = np.array(img)
        tensor = transforms.ToTensor()(img_np)
        tensor = normalize(tensor)
        return img_np, tensor

    # ======================================================================
    # 7c. 收集测试图像
    # ======================================================================
    print('==> Collecting test images..')
    test_images = collect_test_images(args.dataset_path)

    # 统计各模态数量
    n_vis = sum(1 for cam, _, _ in test_images if cam in ('cam1', 'cam2', 'cam4', 'cam5'))
    n_ir = sum(1 for cam, _, _ in test_images if cam in ('cam3', 'cam6'))
    print(f'    Total test images: {len(test_images)}  (VIS: {n_vis}, IR: {n_ir})')

    # ======================================================================
    # 7d. 逐张图像处理
    # ======================================================================
    # 指标累加器
    metric_sums = {'iou': 0.0, 'dice': 0.0, 'recall': 0.0, 'bsr': 0.0, 'acr': 0.0}
    valid_count = 0          # 有效样本计数（有行人区域的）
    skipped_no_mask = 0      # 无分割掩码
    skipped_no_person = 0    # 无行人区域
    total_images = len(test_images)

    start_time = time.time()

    for idx, (cam, pid, img_name) in enumerate(test_images):
        img_path = os.path.join(args.dataset_path, cam, pid, img_name)

        # --- 7d-1. 加载并预处理图像 ---
        img_np, img_tensor = preprocess_image(img_path)
        img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, 3, 384, 144)

        # --- 7d-2. 确定模态 ---
        if cam in ('cam3', 'cam6'):
            modal = 2  # IR
        else:
            modal = 1  # VIS

        # --- 7d-3. 生成 FRL 热图 ---
        heatmap_np = generate_heatmap(net, img_tensor, modal)

        # --- 7d-4. 保存热图 (.npy) 和 Otsu 阈值 ---
        save_dir = os.path.join(args.heatmap_save_path, cam, pid)
        os.makedirs(save_dir, exist_ok=True)

        # 热图文件: 0001.npy
        heatmap_save_path = os.path.join(save_dir, img_name.rsplit('.', 1)[0] + '.npy')
        np.save(heatmap_save_path, heatmap_np.astype(np.float32))

        # --- 7d-5. 计算 Otsu 阈值 ---
        _, otsu_val = otsu_binarize(heatmap_np)

        # Otsu 阈值文件: 0001-otsu.txt
        otsu_save_path = os.path.join(save_dir, img_name.rsplit('.', 1)[0] + '-otsu.txt')
        with open(otsu_save_path, 'w') as f:
            f.write(f'{otsu_val:.6f}')

        # --- 7d-6. 加载行人分割掩码 ---
        mask, has_person = load_mask(args.pedestrian_path, cam, pid, img_name,
                                     target_h=args.img_h, target_w=args.img_w)
        if mask is None:
            skipped_no_mask += 1
            continue
        if not has_person:
            skipped_no_person += 1
            continue

        # --- 7d-7. 计算 5 项指标 ---
        metrics = compute_metrics_single(heatmap_np, mask)

        for key in metric_sums:
            metric_sums[key] += metrics[key]
        valid_count += 1

        # --- 进度打印 ---
        if (idx + 1) % args.batch_size == 0 or (idx + 1) == total_images:
            elapsed = time.time() - start_time
            speed = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total_images - idx - 1) / speed if speed > 0 else 0
            print(f'    [{idx+1}/{total_images}] '
                  f'Elapsed: {elapsed:.1f}s | '
                  f'Speed: {speed:.1f} img/s | '
                  f'ETA: {eta:.1f}s | '
                  f'Valid: {valid_count}')

    # ======================================================================
    # 7e. 汇总与保存
    # ======================================================================
    total_elapsed = time.time() - start_time

    if valid_count == 0:
        print('ERROR: No valid samples! Check dataset paths.')
        return

    # 计算平均值
    avg_metrics = {k: v / valid_count for k, v in metric_sums.items()}

    # 构建输出字符串
    results = []
    results.append('=' * 65)
    results.append('  R1-9: FRL Heatmap Quantitative Evaluation Results')
    results.append('=' * 65)
    results.append(f'')
    results.append(f'  Model checkpoint : {args.checkpoint}')
    results.append(f'  Dataset           : {args.dataset_path}')
    results.append(f'  Pedestrian masks  : {args.pedestrian_path}')
    results.append(f'  Heatmap save dir  : {args.heatmap_save_path}')
    results.append(f'')
    results.append(f'  Total test images       : {total_images}')
    results.append(f'  Valid samples            : {valid_count}')
    results.append(f'  Skipped (no mask file)   : {skipped_no_mask}')
    results.append(f'  Skipped (no person area) : {skipped_no_person}')
    results.append(f'  Processing time          : {total_elapsed:.1f}s')
    results.append(f'')
    results.append(f'  {"Metric":<30} {"Value":>12}')
    results.append(f'  {"-"*30} {"-"*12}')
    results.append(f'  {"mIoU (mean IoU)":<30} {avg_metrics["iou"]:>12.6f}')
    results.append(f'  {"Dice score":<30} {avg_metrics["dice"]:>12.6f}')
    results.append(f'  {"Foreground Recall":<30} {avg_metrics["recall"]:>12.6f}')
    results.append(f'  {"BSR (Background Suppression Ratio)":<30} {avg_metrics["bsr"]:>12.4f}')
    results.append(f'  {"ACR (Attention Concentration Ratio)":<30} {avg_metrics["acr"]:>12.6f}')
    results.append(f'')
    results.append(f'  (mIoU, Dice, Recall are based on Otsu-binarized heatmaps)')
    results.append(f'  (BSR and ACR are based on continuous-value heatmaps)')
    results.append(f'')
    results.append('=' * 65)

    output_str = '\n'.join(results)

    # 打印到控制台
    print('\n' + output_str)

    # 保存到文件（默认保存到当前工作目录，可通过 --results_file 指定路径）
    results_path = args.results_file if os.path.isabs(args.results_file) \
        else os.path.join(os.getcwd(), args.results_file)
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(output_str + '\n')
    print(f'\n==> Results saved to: {results_path}')


if __name__ == '__main__':
    main()
