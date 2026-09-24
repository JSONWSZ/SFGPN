# -*- coding: utf-8 -*-
"""
实验1：身体轮廓多样性度量（SYSU-MM01 vs RegDB，训练集）
======================================================

目的：
  量化两个数据集"同一身份内部"的身体轮廓一致性（类内 mask 两两 IoU），
  并按模态细分为三个指标：可见光内部(VIS-VIS)、红外内部(IR-IR)、
  可见光-红外跨模态(VIS-IR)。通过对比 SYSU-MM01 与 RegDB 的三个
  平均类内 IoU，证明 RegDB 的轮廓多样性显著低于 SYSU-MM01，
  从而支撑论文中"RegDB 姿态/轮廓多样性局限"这一归因
  （回应审稿人第 2 条意见）。

方法：
  - 使用【训练集】身份（SYSU：train_id.txt + val_id.txt 合并；RegDB：trial 1 训练 split）。
  - 对每个身份，按模态把图像分为可见光(VIS)与红外(IR)两组。
  - 每个 mask 缩放到统一尺寸 384x144 并二值化（>0 视为前景）。
  - 对每个身份计算三个类内两两 IoU 的均值：
      * VIS-VIS IoU：可见光图两两之间的 IoU 均值
      * IR-IR  IoU：红外图两两之间的 IoU 均值
      * VIS-IR IoU：可见光 x 红外 跨模态两两之间的 IoU 均值
    IoU 越高 => 同一身份轮廓越接近 => 轮廓多样性越低。
  - 汇总所有身份的三个 IoU，得到每个数据集的三个平均类内 IoU。

预期结论：
  RegDB 的三个平均类内 IoU 均显著高于 SYSU-MM01，说明 RegDB 同一身份
  的轮廓几乎不变（多样性低），从而支持论文的归因。

依赖：
  仅需 numpy + PIL，无需 GPU、无需模型权重。

用法：
  python exp1_silhouette_diversity.py
  运行前请按下方 CONFIG 修改各路径（服务器环境下的实际路径）。

输出：
  - 控制台打印两个数据集的三个指标对比
  - 保存 CSV：silhouette_diversity_results.csv
"""

from __future__ import print_function
import os
import csv
import numpy as np
from PIL import Image
from collections import defaultdict

# ======================================================================
# 可配置参数 —— 在服务器上按实际路径修改
# ======================================================================
CONFIG = {
    # SYSU-MM01 mask 根目录（含 cam1~cam6），结构：{cam}/{id4位}/{id4位}.jpg
    'sysu_mask_root':  '../Datasets/SYSU-MM01-Pedestrian/',

    # SYSU 训练/验证集身份文件（权威来源，与 pre_process_sysu_mask.py 一致）
    # 结构：单行逗号分隔，如 "1,2,3,..."
    # 训练身份 = train_id.txt + val_id.txt 合并
    'sysu_train_id_file': '../Datasets/SYSU-MM01/exp/train_id.txt',
    'sysu_val_id_file':   '../Datasets/SYSU-MM01/exp/val_id.txt',

    # SYSU 相机分模态（标准约定）
    # 可见光相机与红外相机
    'sysu_vis_cams': ['cam1', 'cam2', 'cam4', 'cam5'],
    'sysu_ir_cams':  ['cam3', 'cam6'],

    # RegDB：原图根目录 与 mask 根目录（RegDB-SCHP）
    # mask 结构：RegDB-SCHP/{Visible|Thermal}/{id}/{文件名}（与原图相对路径一致）
    'regdb_mask_root':  '../Datasets/RegDB-SCHP/',

    # RegDB 训练集 split 文件（10 个 trial 共用同一训练集身份 206 id）
    'regdb_train_visible_1': '../Datasets/RegDB/idx/train_visible_1.txt',
    'regdb_train_thermal_1': '../Datasets/RegDB/idx/train_thermal_1.txt',

    # 归一化尺寸（与模型输入对齐）: (height, width)
    'resize': (384, 144),

    # 输出文件
    'output_csv': 'silhouette_diversity_results.csv',
}

# ======================================================================
# 工具函数
# ======================================================================

def load_mask_binary(mask_path, size):
    """
    读取 mask 并二值化后缩放到指定尺寸。
    背景=0，行人>0 -> 二值化为 {0,1}。
    """
    img = Image.open(mask_path).convert('L')
    img = img.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    arr = np.array(img)
    binary = (arr > 0).astype(np.uint8)
    return binary


def mask_iou(a, b):
    """两个二值 mask 的 IoU。"""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0  # 两个都空（理论上不应发生）
    return float(inter) / float(union)


def pairwise_mean_iou(masks_a, masks_b=None):
    """
    计算两组 mask 两两之间的 IoU 均值。
    若 masks_b 为 None，则计算 masks_a 内部两两 IoU（不含自身）。
    返回 (mean_iou, n_pairs)；若样本不足以计算，返回 (None, 0)。
    """
    if masks_b is None:
        masks_b = masks_a
        # 组内两两（不含自身），需要至少 2 个
        if len(masks_a) < 2:
            return None, 0
        ious = []
        for i in range(len(masks_a)):
            for j in range(i + 1, len(masks_a)):
                ious.append(mask_iou(masks_a[i], masks_b[j]))
        return float(np.mean(ious)), len(ious)
    else:
        # 跨组两两，需要两边各至少 1 个
        if len(masks_a) < 1 or len(masks_b) < 1:
            return None, 0
        ious = []
        for a in masks_a:
            for b in masks_b:
                ious.append(mask_iou(a, b))
        return float(np.mean(ious)), len(ious)


def load_masks(paths, size):
    """读取一组 mask 路径为二值数组列表，跳过读取失败的。"""
    masks = []
    for p in paths:
        try:
            masks.append(load_mask_binary(p, size))
        except Exception as e:
            print('  [skip] 无法读取 {}: {}'.format(p, e))
    return masks


def identity_three_iou(vis_paths, ir_paths, size):
    """
    计算单个身份的 VIS-VIS / IR-IR / VIS-IR 三个类内 IoU。
    返回 (vis_vis_iou, ir_ir_iou, vis_ir_iou) 或对应位置 None。
    """
    vis_masks = load_masks(vis_paths, size)
    ir_masks = load_masks(ir_paths, size)

    vv, _ = pairwise_mean_iou(vis_masks)          # 可见光内部
    ii, _ = pairwise_mean_iou(ir_masks)           # 红外内部
    vi, _ = pairwise_mean_iou(vis_masks, ir_masks)  # 跨模态
    return vv, ii, vi


def aggregate(ident_ious):
    """
    汇总所有身份的 [vv, ii, vi]，返回三个 (mean, std, n_valid) 元组。
    ident_ious: list of (vv, ii, vi)，其中每项可能为 None。
    """
    vv_list = [x[0] for x in ident_ious if x[0] is not None]
    ii_list = [x[1] for x in ident_ious if x[1] is not None]
    vi_list = [x[2] for x in ident_ious if x[2] is not None]

    def stat(lst):
        if not lst:
            return (float('nan'), float('nan'), 0)
        return (float(np.mean(lst)), float(np.std(lst)), len(lst))

    return stat(vv_list), stat(ii_list), stat(vi_list)


# ======================================================================
# 各数据集的 mask 路径收集（按身份 + 模态）
# ======================================================================

def collect_sysu_masks(mask_root, train_id_file, val_id_file, vis_cams, ir_cams):
    """
    收集 SYSU 训练/验证集中所有身份的 mask 路径，按身份 + 模态分组。
    训练身份 = train_id.txt + val_id.txt 合并（与 pre_process_sysu_mask.py 一致）。
    返回 {pid: {'vis': [path...], 'ir': [path...]}}
    """
    # 读取训练身份（train_id.txt 与 val_id.txt 合并）
    ids = []
    for f in (train_id_file, val_id_file):
        if not f or not os.path.isfile(f):
            print('[warn] 找不到身份文件: {}'.format(f))
            continue
        with open(f) as fp:
            content = fp.read().strip().splitlines()
        for line in content:
            for x in line.split(','):
                x = x.strip()
                if x:
                    ids.append('%04d' % int(x))
    train_ids = sorted(set(ids))

    grouped = defaultdict(lambda: {'vis': [], 'ir': []})
    for pid in train_ids:
        for cam in vis_cams:
            pid_dir = os.path.join(mask_root, cam, pid)
            if os.path.isdir(pid_dir):
                for fn in sorted(os.listdir(pid_dir)):
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        grouped[pid]['vis'].append(os.path.join(pid_dir, fn))
        for cam in ir_cams:
            pid_dir = os.path.join(mask_root, cam, pid)
            if os.path.isdir(pid_dir):
                for fn in sorted(os.listdir(pid_dir)):
                    if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        grouped[pid]['ir'].append(os.path.join(pid_dir, fn))

    # 过滤掉既无 vis 也无 ir 的身份
    grouped = {k: v for k, v in grouped.items() if v['vis'] or v['ir']}
    return grouped


def collect_regdb_masks(mask_root, train_visible_file, train_thermal_file):
    """
    收集 RegDB 训练集中所有身份的 mask 路径，按身份 + 模态分组。
    txt 格式：'{Visible|Thermal}/{id}/{file} {label}'，label 从 0 开始。
    mask 相对路径与原图一致，只需把根目录换成 mask_root。
    返回 {label: {'vis': [path...], 'ir': [path...]}}
    """
    grouped = defaultdict(lambda: {'vis': [], 'ir': []})
    for split_file, modal in ((train_visible_file, 'vis'),
                              (train_thermal_file, 'ir')):
        if not os.path.isfile(split_file):
            print('[warn] 找不到 {}'.format(split_file))
            continue
        with open(split_file) as fp:
            lines = fp.read().splitlines()
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(' ')
            rel_path = parts[0]            # e.g. Visible/285/xxx.bmp
            label = int(parts[1])
            mask_path = os.path.join(mask_root, rel_path)
            grouped[label][modal].append(mask_path)
    return grouped


# ======================================================================
# 主流程
# ======================================================================

def process_one_dataset(name, groups, size):
    """对一组 {pid: {'vis':[], 'ir':[]}} 计算三个指标的聚合结果。"""
    ident_ious = []
    for pid, g in groups.items():
        vv, ii, vi = identity_three_iou(g['vis'], g['ir'], size)
        ident_ious.append((vv, ii, vi))
    vv_stat, ii_stat, vi_stat = aggregate(ident_ious)
    return vv_stat, ii_stat, vi_stat


def main():
    size = CONFIG['resize']
    results = {}

    # ---- SYSU-MM01 ----
    print('=' * 64)
    print('处理 SYSU-MM01 (训练集) ...')
    sysu_groups = collect_sysu_masks(
        CONFIG['sysu_mask_root'],
        CONFIG['sysu_train_id_file'],
        CONFIG['sysu_val_id_file'],
        CONFIG['sysu_vis_cams'],
        CONFIG['sysu_ir_cams'],
    )
    vv, ii, vi = process_one_dataset('SYSU-MM01', sysu_groups, size)
    results['SYSU-MM01'] = (vv, ii, vi)
    print('  身份数={}'.format(len(sysu_groups)))
    print('  VIS-VIS IoU: mean={:.4f} std={:.4f} n={}'.format(*vv))
    print('  IR-IR   IoU: mean={:.4f} std={:.4f} n={}'.format(*ii))
    print('  VIS-IR  IoU: mean={:.4f} std={:.4f} n={}'.format(*vi))

    # ---- RegDB ----
    print('=' * 64)
    print('处理 RegDB (训练集) ...')
    regdb_groups = collect_regdb_masks(
        CONFIG['regdb_mask_root'],
        CONFIG['regdb_train_visible_1'],
        CONFIG['regdb_train_thermal_1'],
    )
    vv, ii, vi = process_one_dataset('RegDB', regdb_groups, size)
    results['RegDB'] = (vv, ii, vi)
    print('  身份数={}'.format(len(regdb_groups)))
    print('  VIS-VIS IoU: mean={:.4f} std={:.4f} n={}'.format(*vv))
    print('  IR-IR   IoU: mean={:.4f} std={:.4f} n={}'.format(*ii))
    print('  VIS-IR  IoU: mean={:.4f} std={:.4f} n={}'.format(*vi))

    # ---- 汇总 ----
    print('=' * 64)
    print('结果汇总（平均类内 IoU，越高 => 轮廓越一致 => 多样性越低）：')
    print('  {:<12s}  {:>10s}  {:>10s}  {:>10s}'.format('数据集', 'VIS-VIS', 'IR-IR', 'VIS-IR'))
    for name in ['SYSU-MM01', 'RegDB']:
        vv, ii, vi = results[name]
        print('  {:<12s}  {:>10.4f}  {:>10.4f}  {:>10.4f}'.format(
            name, vv[0], ii[0], vi[0]))

    # 结论提示
    print('-' * 64)
    svv, sii, svi = results['SYSU-MM01']
    rvv, rii, rvi = results['RegDB']
    diffs = {
        'VIS-VIS': rvv[0] - svv[0],
        'IR-IR': rii[0] - sii[0],
        'VIS-IR': rvi[0] - svi[0],
    }

    print('RegDB 与 SYSU-MM01 的差值（RegDB - SYSU）：')
    for k, d in diffs.items():
        print('  {}: {:+.4f}'.format(k, d))
    all_positive = all(d > 0 for d in diffs.values())
    if all_positive:
        print('=> RegDB 在三个模态维度上的轮廓一致性均更高，轮廓多样性显著更低，')
        print('   支持论文"RegDB 姿态/轮廓多样性局限"的归因。')
    else:
        print('=> 注意：并非所有维度都符合预期，请检查数据或重新审视归因。')

    # ---- 保存 CSV ----
    out_csv = CONFIG['output_csv']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'dataset', 'n_ids',
            'vis_vis_iou_mean', 'vis_vis_iou_std', 'vis_vis_n',
            'ir_ir_iou_mean', 'ir_ir_iou_std', 'ir_ir_n',
            'vis_ir_iou_mean', 'vis_ir_iou_std', 'vis_ir_n',
        ])
        n_ids = {'SYSU-MM01': len(sysu_groups), 'RegDB': len(regdb_groups)}
        for name in ['SYSU-MM01', 'RegDB']:
            vv, ii, vi = results[name]
            writer.writerow([
                name, n_ids[name],
                vv[0], vv[1], vv[2],
                ii[0], ii[1], ii[2],
                vi[0], vi[1], vi[2],
            ])
    print('结果已保存到: {}'.format(out_csv))


if __name__ == '__main__':
    main()
