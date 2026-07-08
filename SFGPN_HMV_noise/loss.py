import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        # dist.addmm_(1, -2, inputs, inputs.t())
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct



        
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    # dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx



class FocalLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(FocalLoss, self).__init__()
        self.smooth = smooth

    def forward(self, mask, heatmaps):
        """
        pedestrian: [B, 3, H, W] 去背景后的原图（背景像素=0）
        heatmaps:   [B, 1, H, W]  输出的热力图（Sigmoid 后）
        """

        # ==== 尺寸检查 ====
        assert mask.shape[2:] == (384, 144), \
            f"Mask size mismatch: got {mask.shape[2:]}, expected (384, 144)"
        assert heatmaps.shape[2:] == (384, 144), \
            f"Heatmap size mismatch: got {heatmaps.shape[2:]}, expected (384, 144)"

        # BCE Loss
        bce = F.binary_cross_entropy(heatmaps, mask)

        # Dice Loss
        intersection = (heatmaps * mask).sum()
        dice = 1 - (2. * intersection + self.smooth) / \
               (heatmaps.sum() + mask.sum() + self.smooth)

        # 总损失
        loss = bce + dice
        return loss

def modality_center_loss(feats, labels):
    """
    红外-可见光多中心约束损失函数 (改进版: 使用余弦相似度计算中心, 不归一化输入特征)

    Args:
        feats: [2B, dim] 特征向量，前B为可见光，后B为红外
        labels: [2B]      身份id

    Returns:
        owCompact_loss, R_loss, twCompact_loss, HC_loss, LCA_loss
    """
    device = feats.device
    N, dim = feats.size()
    B = N // 2

    # 划分模态
    feats_v=feats[:B]
    feats_ir=feats[B:2*B]

    labels_v=labels[:B]
    labels_ir=labels[B:2*B]
    unique_ids = labels.unique()
    P = len(unique_ids)

    # ==================================================
    # 权重身份中心计算函数 (使用余弦相似度，避免数值爆炸)
    # ==================================================
    def weighted_center(feats_grp):
        """
        feats_grp: [K, dim] 该身份在某模态下的全部特征
        return: [dim] 该身份的加权中心
        """
        K, dim = feats_grp.shape
        if K == 1:
            return feats_grp.squeeze(0)

        # ---------- 计算余弦相似度矩阵 ----------
        normed = F.normalize(feats_grp, p=2, dim=1)              # 用归一化特征算相似度
        cos_sim = torch.exp(normed @ normed.t())                 # [K, K]
        diag = torch.diag(cos_sim)                               # 自相似度=1

        # ---------- 计算权重 ----------
        denom = cos_sim.sum() - diag.sum()                       # 排除自对角线
        numer = cos_sim.sum(dim=1) - diag
        weights = numer / denom

        # ---------- 用未归一化特征加权求中心 ----------
        center = (weights.unsqueeze(1) * feats_grp).sum(dim=0)
        return center

   
    # ==================================================
    # 2. 分别计算可见光 / 红外 模态中心
    # ==================================================
    centers_v = torch.stack(
        [weighted_center(feats_v[labels_v == pid]) for pid in unique_ids], dim=0
    )
    centers_ir = torch.stack(
        [weighted_center(feats_ir[labels_ir == pid]) for pid in unique_ids], dim=0
    )

    # ==================================================
    # D. HC_loss: 双模态中心互约束 (中心 ↔ 中心)
    # ==================================================
    HC_loss = torch.norm(centers_v - centers_ir, dim=1).mean()

    # ==================================================
    # C. twCompact_loss: 双模态中心紧凑性 (样本 → 各自模态中心)
    # ==================================================
    twCompact_loss = 0.0
    for i, pid in enumerate(unique_ids):
        if (labels_v == pid).any():
            dists_v = torch.norm(feats_v[labels_v == pid] - centers_v[i], dim=1)
            twCompact_loss += dists_v.mean()
        if (labels_ir == pid).any():
            dists_ir = torch.norm(feats_ir[labels_ir == pid] - centers_ir[i], dim=1)
            twCompact_loss += dists_ir.mean()
    twCompact_loss = twCompact_loss / (2 * P)

    return HC_loss,twCompact_loss
