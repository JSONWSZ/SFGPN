# -*- coding: utf-8 -*-
"""
实验3：跨模态轮廓相似度分层评估（RegDB / SYSU，IR→VIS）
=================================================

目的：
  检验"RegDB 训练的模型过度依赖体型/轮廓特征做判别"这一观点。
  分档依据是"query 与同身份 gallery 的平均轮廓 IoU"（跨模态轮廓对齐度）：

  - 轮廓对齐 IoU 高：同一身份的 IR/可见光轮廓几乎一致，轮廓变化小；
  - 轮廓对齐 IoU 低：同一身份的 IR/可见光轮廓差异大，轮廓变化大。

  若模型依赖体型/轮廓特征做判别，则：
    当同一身份的轮廓变化小（IoU 高）时，训练信号充分，模型判别稳定，表现好；
    当同一身份的轮廓变化大（IoU 低）时，模型难以匹配，表现差。
    因此"高轮廓对齐档"应好于"低轮廓对齐档"（差值方向：高 > 低）。

方法：
  1. 加载 SFGPN 模型权重，提取 query（thermal）与 gallery（visible）特征。
  2. 对每个 query，计算它与所有【不同身份】gallery 的 mask IoU 均值，
     作为该 query 的"轮廓大众化程度"（与负例的相似度）。
  3. 按 IoU 中位数二分高低两档，分别用官方 eval 函数统计 Rank-1 / mAP。
  4. 输出两档指标及其差值。

加速：
  所有 mask 一次性缩放到统一尺寸并展平为向量，用 torch 矩阵运算
  （query_mask @ gallery_mask^T）在 GPU 上批量计算两两 IoU，
  避免 Python 双重循环，速度提升 2~3 个数量级。

用法：
  RegDB: python exp3_stratified_eval.py --dataset regdb --gpu 0 --all-trials
  SYSU:  python exp3_stratified_eval.py --dataset sysu --gpu 0 --resume HC04_twCompact002-65_150.t

依赖：
  需要 GPU + 已训练模型权重；脚本须放在 SFGPN 目录下运行
  （需要 import data_manager / data_loader / model / eval_metrics）。
"""

from __future__ import print_function
import os
import csv
import argparse
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from data_loader import TestData
from data_manager import (process_test_regdb,
                          process_query_sysu, process_gallery_sysu)
from model import embed_net
from eval_metrics import eval_regdb, eval_sysu

# ======================================================================
# 可配置参数 —— 在服务器上按实际路径修改
# ======================================================================
CONFIG = {
    # RegDB 数据根目录与 mask 根目录
    'regdb_data_root': '../Datasets/RegDB/',
    'regdb_mask_root': '../Datasets/RegDB-SCHP/',

    # SYSU 数据根目录与 mask 根目录
    'sysu_data_root': '../Datasets/SYSU-MM01/',
    'sysu_mask_root': '../Datasets/SYSU-MM01-Pedestrian/',

    # RegDB checkpoint 目录与命名规则（与 test_query_RegDB.py 一致）
    'regdb_checkpoint_dir': 'save_model/',
    'regdb_checkpoint_pattern': 'regdb_agw_p4_n6_lr_0.1_seed_0_trial_{trial}_best.t',

    # SYSU checkpoint 目录（用 --resume 指定文件名）
    'sysu_checkpoint_dir': 'save_model/',

    # 模型结构参数（RegDB n_class=206，SYSU n_class=395）
    'regdb_n_class': 206,
    'sysu_n_class': 395,
    'arch': 'resnet50',
    'img_w': 144,
    'img_h': 384,
    'pool_dim': 2048,

    # mask 归一化尺寸（与模型输入对齐）: (height, width)
    'mask_size': (384, 144),

    # 输出
    'output_csv': 'stratified_eval_results.csv',
}

# ======================================================================
# mask 工具函数
# ======================================================================

def load_mask_binary(mask_path, size):
    """读取 mask，二值化并缩放到统一尺寸。背景=0，行人>0。"""
    img = Image.open(mask_path).convert('L')
    img = img.resize((size[1], size[0]), Image.Resampling.LANCZOS)
    arr = np.array(img)
    return (arr > 0).astype(np.uint8)


def mask_path_from_image(img_path, dataset, mask_root):
    """
    由原图路径推导 mask 路径。
    RegDB:  原图 .../RegDB/{Visible|Thermal}/{id}/{file}
            mask  .../RegDB-SCHP/{Visible|Thermal}/{id}/{file}
    SYSU:   原图 .../SYSU-MM01/{cam}/{id}/{file}
            mask  .../SYSU-MM01-Pedestrian/{cam}/{id}/{file}
    """
    parts = img_path.replace('\\', '/').split('/')
    if dataset == 'regdb':
        idx = None
        for i, p in enumerate(parts):
            if p in ('Visible', 'Thermal'):
                idx = i
                break
        if idx is None:
            raise ValueError('无法定位 Visible/Thermal: {}'.format(img_path))
        rel = '/'.join(parts[idx:])
        return os.path.join(mask_root, rel)
    else:  # sysu
        # 形如 .../SYSU-MM01/cam3/0012/0012.jpg
        # 取最后三段 {cam}/{id}/{file} 拼到 mask_root 后
        rel = '/'.join(parts[-3:])
        return os.path.join(mask_root, rel)


def compute_iou_matrix(q_masks, g_masks, device):
    """
    用 torch 批量计算 query x gallery 的两两 IoU 矩阵。
    q_masks/g_masks: list of uint8 二值数组，尺寸一致 (H, W)。
    返回 shape (nq, ng) 的 float32 tensor（CPU）。
    将 mask 展平为向量后做矩阵乘法，利用 GPU 加速。
    """
    H, W = q_masks[0].shape
    D = H * W
    Q = np.stack([m.reshape(-1) for m in q_masks]).astype(np.float32)   # (nq, D)
    G = np.stack([m.reshape(-1) for m in g_masks]).astype(np.float32)   # (ng, D)
    Q_t = torch.from_numpy(Q).to(device)
    G_t = torch.from_numpy(G).to(device)

    q_area = Q_t.sum(dim=1)                     # (nq,)
    g_area = G_t.sum(dim=1)                     # (ng,)
    inter = Q_t @ G_t.t()                       # (nq, ng) 交集像素数
    union = q_area[:, None] + g_area[None, :] - inter
    iou = inter / union.clamp(min=1e-6)
    return iou.cpu().numpy()


# ======================================================================
# 分层评估
# ======================================================================

def stratified_metrics(distmat, q_pids, g_pids, contour_ious,
                       q_camids=None, g_camids=None):
    """
    按轮廓对齐 IoU（query 与同身份 gallery 的平均 mask IoU）中位数二分，
    分别用官方 eval 函数统计两档的 Rank-1 与 mAP。
    distmat: (nquery, ngall) 相似度矩阵（值越大越相似）。
    contour_ious: 每个 query 的轮廓对齐 IoU（与 distmat 行对应）。
    """
    ious = np.asarray(contour_ious, dtype=np.float64)
    median = float(np.median(ious))
    high_idx = np.where(ious >= median)[0]
    low_idx = np.where(ious < median)[0]

    def eval_subset(idxs):
        if len(idxs) == 0:
            return (0.0, 0.0, 0)
        sub_dist = distmat[idxs]
        sub_q = q_pids[idxs]
        if q_camids is not None and g_camids is not None:
            # SYSU：带相机约束
            cmc, mAP, mINP = eval_sysu(-sub_dist, sub_q, g_pids,
                                       q_camids[idxs], g_camids)
        else:
            cmc, mAP, mINP = eval_regdb(-sub_dist, sub_q, g_pids)
        return (float(cmc[0]), float(mAP), len(idxs))

    high = eval_subset(high_idx)
    low = eval_subset(low_idx)
    return {'median_iou': median, 'high': high, 'low': low}


# ======================================================================
# 主流程
# ======================================================================

def run_one(dataset, trial, net, device, transform_test, mode='all'):
    """对单个数据集/trial 执行分层评估，返回结果 dict。"""
    if dataset == 'regdb':
        data_root = CONFIG['regdb_data_root']
        mask_root = CONFIG['regdb_mask_root']
        query_img, query_label = process_test_regdb(data_root, trial=trial, modal='thermal')
        gall_img, gall_label = process_test_regdb(data_root, trial=trial, modal='visible')
        q_camids = None
        g_camids = None
    else:  # sysu
        data_root = CONFIG['sysu_data_root']
        mask_root = CONFIG['sysu_mask_root']
        query_img, query_label, query_cam = process_query_sysu(data_root, mode=mode)
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_root, mode=mode, trial=trial)
        q_camids = query_cam
        g_camids = gall_cam

    gallset = TestData(gall_img, gall_label, transform=transform_test,
                       img_size=(CONFIG['img_w'], CONFIG['img_h']))
    queryset = TestData(query_img, query_label, transform=transform_test,
                        img_size=(CONFIG['img_w'], CONFIG['img_h']))

    gall_loader = data.DataLoader(gallset, batch_size=32, shuffle=False, num_workers=4)
    query_loader = data.DataLoader(queryset, batch_size=32, shuffle=False, num_workers=4)

    nquery = len(query_label)
    ngall = len(gall_label)

    # ---- 提取特征 ----
    def extract(loader, n, modal):
        feat1 = np.zeros((n, CONFIG['pool_dim']))
        feat2 = np.zeros((n, CONFIG['pool_dim']))
        feat3 = np.zeros((n, CONFIG['pool_dim']))
        att1 = np.zeros((n, CONFIG['pool_dim']))
        att2 = np.zeros((n, CONFIG['pool_dim']))
        att3 = np.zeros((n, CONFIG['pool_dim']))
        ptr = 0
        with torch.no_grad():
            for inp, _ in loader:
                bn = inp.size(0)
                inp = inp.to(device)
                feat, feat_att = net(inp, inp, modal=modal)
                feat1[ptr:ptr + bn] = feat[:bn].detach().cpu().numpy()
                feat2[ptr:ptr + bn] = feat[bn:2 * bn].detach().cpu().numpy()
                feat3[ptr:ptr + bn] = feat[2 * bn:3 * bn].detach().cpu().numpy()
                att1[ptr:ptr + bn] = feat_att[:bn].detach().cpu().numpy()
                att2[ptr:ptr + bn] = feat_att[bn:2 * bn].detach().cpu().numpy()
                att3[ptr:ptr + bn] = feat_att[2 * bn:3 * bn].detach().cpu().numpy()
                ptr += bn
        return (feat1, feat2, feat3), (att1, att2, att3)

    print('  提取 query 特征 (thermal, modal=2)...')
    qf, qa = extract(query_loader, nquery, modal=2)
    print('  提取 gallery 特征 (visible, modal=1)...')
    gf, ga = extract(gall_loader, ngall, modal=1)

    distmat = (np.matmul(qf[0], gf[0].T) + np.matmul(qf[1], gf[1].T) + np.matmul(qf[2], gf[2].T)) / 3
    distmat_att = (np.matmul(qa[0], ga[0].T) + np.matmul(qa[1], ga[1].T) + np.matmul(qa[2], ga[2].T)) / 3
    if dataset == 'regdb':
        a = 0.3
    else:
        a = 0.5  # SYSU 用 (pool + att)/2
    distmat_all = a * distmat + (1 - a) * distmat_att

    # ---- 加载所有 mask 并计算 IoU 矩阵 ----
    print('  加载 query/gallery mask 并计算 IoU 矩阵 (GPU 加速)...')
    size = CONFIG['mask_size']
    q_masks = []
    for p in query_img:
        try:
            q_masks.append(load_mask_binary(mask_path_from_image(p, dataset, mask_root), size))
        except Exception as e:
            print('    [warn] query mask 失败 {}: {}'.format(p, e))
    g_masks = []
    for p in gall_img:
        try:
            g_masks.append(load_mask_binary(mask_path_from_image(p, dataset, mask_root), size))
        except Exception as e:
            print('    [warn] gallery mask 失败 {}: {}'.format(p, e))

    # 检查是否有 mask 加载失败（失败会打乱索引对应，这里要求全部成功）
    assert len(q_masks) == nquery, '部分 query mask 加载失败，无法对齐'
    assert len(g_masks) == ngall, '部分 gallery mask 加载失败，无法对齐'

    iou_mat = compute_iou_matrix(q_masks, g_masks, device)   # (nq, ng)

    # ---- 分档 IoU：query 与所有"同身份"gallery 的平均 IoU ----
    # 度量"该身份的跨模态轮廓对齐程度"（红外 query 与可见光同身份 gallery 的轮廓相似度）：
    #   IoU 高 => 同一身份的 IR/可见光轮廓对齐好，轮廓变化小；
    #   IoU 低 => 同一身份的 IR/可见光轮廓对齐差，轮廓变化大。
    # 对每个 query，取其正例 gallery（label 相同）对应列的 IoU 求平均。
    q_lbl = np.asarray(query_label, dtype=np.int64)
    g_lbl = np.asarray(gall_label, dtype=np.int64)
    same_label = (q_lbl[:, None] == g_lbl[None, :])           # (nq, ng) 同身份掩码
    pos_ious = np.where(same_label, iou_mat, 0.0)             # 不同身份置 0
    pos_cnt = same_label.sum(axis=1)                          # 每个 query 的正例数
    # 防止除零：无正例的 query 会得到 0（后续可用 pos_cnt 过滤）
    contour_ious = np.where(pos_cnt > 0,
                            pos_ious.sum(axis=1) / np.maximum(pos_cnt, 1),
                            0.0)

    res = stratified_metrics(distmat_all, query_label, gall_label, contour_ious,
                             q_camids, g_camids)
    res['n_query'] = nquery
    res['n_gallery'] = ngall
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='regdb', type=str,
                        choices=['regdb', 'sysu'])
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--trial', default=1, type=int,
                        help='RegDB trial 编号 / SYSU gallery 采样 seed')
    parser.add_argument('--all-trials', action='store_true',
                        help='遍历 RegDB trial 1~10 并求平均（仅 regdb 有效）')
    parser.add_argument('--resume', default='', type=str,
                        help='SYSU checkpoint 文件名（如 HC04_twCompact002-65_150.t）')
    parser.add_argument('--mode', default='all', type=str, choices=['all', 'indoor'])
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['img_h'], CONFIG['img_w'])),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = args.dataset
    n_class = CONFIG['regdb_n_class'] if dataset == 'regdb' else CONFIG['sysu_n_class']

    print('==> 构建模型 (n_class={})...'.format(n_class))
    net = embed_net(n_class, gm_pool='off', arch=CONFIG['arch'])
    net.to(device)
    cudnn.benchmark = True

    if dataset == 'regdb':
        trials = range(1, 11) if args.all_trials else [args.trial]
        all_results = []
        for trial in trials:
            model_path = os.path.join(
                CONFIG['regdb_checkpoint_dir'],
                CONFIG['regdb_checkpoint_pattern'].format(trial=trial))
            if not os.path.isfile(model_path):
                print('[warn] 找不到 checkpoint: {}，跳过 trial {}'.format(model_path, trial))
                continue
            print('==> 加载 checkpoint: {}'.format(model_path))
            checkpoint = torch.load(model_path, weights_only=False)
            net.load_state_dict(checkpoint['net'])
            net.eval()

            print('==> 处理 RegDB trial {} ...'.format(trial))
            res = run_one(dataset, trial, net, device, transform_test, args.mode)
            res['trial'] = trial
            all_results.append(res)

            print('    trial {}: median IoU={:.4f}'.format(trial, res['median_iou']))
            print('      高轮廓对齐 (IoU>={:.4f}): Rank-1={:.2%}, mAP={:.2%}, n={}'.format(
                res['median_iou'], res['high'][0], res['high'][1], res['high'][2]))
            print('      低轮廓对齐 (IoU< {:.4f}): Rank-1={:.2%}, mAP={:.2%}, n={}'.format(
                res['median_iou'], res['low'][0], res['low'][1], res['low'][2]))

        if not all_results:
            print('没有可用的 trial 结果。')
            return

        print('=' * 70)
        print('RegDB 汇总（{} 个 trial）：'.format(len(all_results)))
        avg_median_iou = np.mean([r['median_iou'] for r in all_results])
        high_r1 = np.mean([r['high'][0] for r in all_results])
        high_map = np.mean([r['high'][1] for r in all_results])
        high_n = np.mean([r['high'][2] for r in all_results])
        low_r1 = np.mean([r['low'][0] for r in all_results])
        low_map = np.mean([r['low'][1] for r in all_results])
        low_n = np.mean([r['low'][2] for r in all_results])
        print('  平均中位数 IoU 阈值 = {:.4f}'.format(avg_median_iou))
        print('  高轮廓对齐档 (IoU>=median): Rank-1={:.2%}, mAP={:.2%}, 平均 n={:.1f}'.format(
            high_r1, high_map, high_n))
        print('  低轮廓对齐档 (IoU< median): Rank-1={:.2%}, mAP={:.2%}, 平均 n={:.1f}'.format(
            low_r1, low_map, low_n))
        print('  差值 (高-低): Rank-1={:+.2%}, mAP={:+.2%}'.format(
            high_r1 - low_r1, high_map - low_map))

        with open(CONFIG['output_csv'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'trial', 'median_iou',
                             'high_rank1', 'high_map', 'high_n',
                             'low_rank1', 'low_map', 'low_n'])
            for r in all_results:
                writer.writerow([dataset, r['trial'], r['median_iou'],
                                 r['high'][0], r['high'][1], r['high'][2],
                                 r['low'][0], r['low'][1], r['low'][2]])
        print('结果已保存到: {}'.format(CONFIG['output_csv']))

    else:  # sysu
        if not args.resume:
            print('[error] SYSU 需要 --resume 指定 checkpoint 文件名')
            return
        model_path = os.path.join(CONFIG['sysu_checkpoint_dir'], args.resume)
        if not os.path.isfile(model_path):
            print('[error] 找不到 checkpoint: {}'.format(model_path))
            return
        print('==> 加载 checkpoint: {}'.format(model_path))
        checkpoint = torch.load(model_path, weights_only=False)
        net.load_state_dict(checkpoint['net'])
        net.eval()

        print('==> 处理 SYSU (trial={}, mode={}) ...'.format(args.trial, args.mode))
        res = run_one(dataset, args.trial, net, device, transform_test, args.mode)

        print('=' * 70)
        print('SYSU 结果:')
        print('  median IoU={:.4f}'.format(res['median_iou']))
        print('  高轮廓对齐 (IoU>={:.4f}): Rank-1={:.2%}, mAP={:.2%}, n={}'.format(
            res['median_iou'], res['high'][0], res['high'][1], res['high'][2]))
        print('  低轮廓对齐 (IoU< {:.4f}): Rank-1={:.2%}, mAP={:.2%}, n={}'.format(
            res['median_iou'], res['low'][0], res['low'][1], res['low'][2]))
        print('  差值 (高-低): Rank-1={:+.2%}, mAP={:+.2%}'.format(
            res['high'][0] - res['low'][0], res['high'][1] - res['low'][1]))

        with open(CONFIG['output_csv'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'trial', 'median_iou',
                             'high_rank1', 'high_map', 'high_n',
                             'low_rank1', 'low_map', 'low_n'])
            writer.writerow([dataset, args.trial, res['median_iou'],
                             res['high'][0], res['high'][1], res['high'][2],
                             res['low'][0], res['low'][1], res['low'][2]])
        print('结果已保存到: {}'.format(CONFIG['output_csv']))


if __name__ == '__main__':
    main()
