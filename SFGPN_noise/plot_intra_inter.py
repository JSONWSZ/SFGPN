import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from model import embed_net   # 复用你的工程

########################################
# 配置区
########################################
MODEL_PATH = "save_model/sysu_agw_p4_n6_lr_0.1_seed_0_best.t"
SYSU_PATH = "/media/ssd_2t/home/wsz/python/projects/Datasets/SYSU-MM01/"

# 建议：为了画出更平滑的分布，建议稍微增加 ID 数量，或者 K 值
# DEEN 论文是从测试集随机选了 20 个 ID
selected_ids = [89,90,93,102,105,108,112,116,117,122,
                125,129,130,134,138,139,150,152,162,166]

K = 10  # 每个模态 K 张图

vis_cams = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cams  = ['cam3', 'cam6']

IMG_W = 144
IMG_H = 384

device = 'cuda:3'

########################################
# 预处理
########################################
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

########################################
# 加载模型
########################################
n_class = 395
net = embed_net(n_class, gm_pool='off', arch='resnet50', dataset='sysu')
net.to(device)
cudnn.benchmark = True

print("Loading model:", MODEL_PATH)
checkpoint = torch.load(MODEL_PATH, weights_only=False)
net.load_state_dict(checkpoint["net"])
net.eval()

########################################
# 工具函数
########################################
def load_images_for_id(pid, cams, base_path):
    img_list_per_cam = {}
    for cam in cams:
        cam_dir = os.path.join(base_path, cam, "{:04d}".format(pid))
        if not os.path.isdir(cam_dir):
            continue
        images = sorted([os.path.join(cam_dir, x)
                         for x in os.listdir(cam_dir)
                         if x.lower().endswith(".jpg")])
        if len(images) > 0:
            img_list_per_cam[cam] = images
    return img_list_per_cam

def pick_evenly_from_cams(img_lists, K):
    cams = list(img_lists.keys())
    C = len(cams)
    if C == 0:
        return []
    base = K // C
    remain = K - base * C
    selected = []
    for idx, cam in enumerate(cams):
        take = base + (1 if idx < remain else 0)
        imgs = img_lists[cam][:take]
        selected.extend(imgs)
    return selected

def extract_embedding(img_path, modal=1):
    img = Image.open(img_path).convert("RGB")
    img = np.array(img)
    img = transform_test(img)
    img = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        feat, feat_att = net(img, img, modal=modal)
        
    f1 = feat.mean(dim=0)
    f2 = feat_att.mean(dim=0)
    
    a = 0.5
    final_feat = a*f1 + (1-a)*f2
    
    # 既然你确认模型内部已经做了 L2 Norm，这里就不重复做了
    # 但为了保险，通常建议 output / output.norm()
    final_feat = final_feat / final_feat.norm(p=2)
    return final_feat.cpu().numpy()

########################################
# 主流程：数据收集
########################################
vis_feats_list = []
vis_labels_list = []

ir_feats_list = []
ir_labels_list = []

print("Extracting features...")
for idx, pid in enumerate(selected_ids):
    # VIS
    vis_lists = load_images_for_id(pid, vis_cams, SYSU_PATH)
    vis_imgs = pick_evenly_from_cams(vis_lists, K)
    for path in vis_imgs:
        feat = extract_embedding(path, modal=1)
        vis_feats_list.append(feat)
        vis_labels_list.append(pid) # 使用真实 PID 或 enumerate ID 均可，只要匹配即可

    # IR
    ir_lists = load_images_for_id(pid, ir_cams, SYSU_PATH)
    ir_imgs = pick_evenly_from_cams(ir_lists, K)
    for path in ir_imgs:
        feat = extract_embedding(path, modal=2)
        ir_feats_list.append(feat)
        ir_labels_list.append(pid)

# 转换为 numpy 矩阵
vis_feats = np.array(vis_feats_list) # (N_vis, Dim)
ir_feats = np.array(ir_feats_list)   # (N_ir, Dim)
vis_labels = np.array(vis_labels_list)
ir_labels = np.array(ir_labels_list)

print(f"Features extracted. VIS: {vis_feats.shape}, IR: {ir_feats.shape}")

########################################
# 核心修改：矩阵化计算 Cosine Distance
########################################

# 1. 计算相似度矩阵 (Cosine Similarity)
# 因为特征已归一化，所以 dot product 就是 cosine similarity
# Shape: (N_vis, N_ir)
sim_mat = np.matmul(vis_feats, ir_feats.T)

# 2. 转换为 Cosine Distance
# Distance = 1 - Similarity
dist_mat = 1.0 - sim_mat

# 3. 创建 Mask 来区分 Intra (同ID) 和 Inter (不同ID)
# vis_labels[:, None] 形状是 (N_vis, 1)
# ir_labels[None, :] 形状是 (1, N_ir)
# 广播后生成 bool 矩阵 (N_vis, N_ir)
mask = (vis_labels[:, None] == ir_labels[None, :])

# 4. 利用 Mask 提取距离
intra = dist_mat[mask]      # 所有的正样本对距离
inter = dist_mat[~mask]     # 所有的负样本对距离

print(f"Intra samples: {len(intra)}, Inter samples: {len(inter)}")

########################################
# 画图：DEEN 风格复刻
########################################
# 设置字体大小，模仿论文风格
plt.rcParams.update({'font.size': 14})

fig, ax = plt.subplots(figsize=(12, 10), dpi=120)

# 设置 bins 范围，Cosine Distance 理论是 [0, 2]，通常分布在 0 到 1.2 左右
# 根据你的数据分布，可以微调 num
bins = 150

# 1. 画直方图
# 关键参数: density=True (概率密度), histtype='stepfilled' (填充风格), alpha=0.5 (透明度)
ax.hist(intra, bins, histtype="stepfilled", alpha=0.5, color='blue', density=True, label='Intra-class')
ax.hist(inter, bins, histtype="stepfilled", alpha=0.5, color='green', density=True, label='Inter-class')

# 2. 计算均值
mean_intra = np.mean(intra)
mean_inter = np.mean(inter)
d_val = mean_inter - mean_intra

# 3. 获取 Y 轴高度以便放置文字
# 获取当前坐标轴的限制，确保文字画在图内
y_min, y_max = ax.get_ylim()
text_y_pos = y_max * 0.85  # 文字高度
arrow_y_pos = y_max * 0.82  # 箭头高度

# ================= 替换开始 =================

# 4. 画均值虚线
# 这里的 y_max 取决于直方图的高度，自动获取
y_min, y_max = ax.get_ylim()

# 绘制虚线
ax.axvline(mean_intra, color='blue', linestyle='--', linewidth=2, alpha=0.8)
ax.axvline(mean_inter, color='green', linestyle='--', linewidth=2, alpha=0.8)

# ---------------------------------------------------------------
# 新增需求 A & B：在竖线旁边标注均值大小
# ---------------------------------------------------------------
# 设置均值文字的高度，建议在箭头的下方一点点，或者在峰值附近
mean_text_y_pos = y_max * 0.6  

# Intra 均值：标在蓝线右侧 (ha='left')
ax.text(mean_intra + 0.01, mean_text_y_pos, f"{mean_intra:.4f}", 
        color='blue', ha='left', va='center', fontweight='bold')

# Inter 均值：标在绿线左侧 (ha='right')
ax.text(mean_inter - 0.01, mean_text_y_pos, f"{mean_inter:.4f}", 
        color='green', ha='right', va='center', fontweight='bold')

# ---------------------------------------------------------------
# 新增需求 C：Delta 符号加上下标
# ---------------------------------------------------------------
mid_point = (mean_intra + mean_inter) / 2
arrow_y_pos = y_max * 0.82  # 箭头高度
text_y_pos = y_max * 0.85   # Delta文字高度

# 使用 LaTeX 语法 _{} 来添加下标
# 这里的 delta_subscript 是你在上面定义的变量，比如 "5"
delta_subscript = "5"  # 这里填你想要的数字，比如 "5" 会显示为 δ₅
delta_str = fr"$\delta_{{{delta_subscript}}} = {d_val:.4f}$"

# 画中间的 Delta 文字
ax.text(mid_point, text_y_pos, delta_str, color='black', 
        ha='center', va='bottom', fontsize=18, fontweight='bold')

# 画双向箭头
# 左半边箭头：从中间指向 Intra
ax.annotate('', xy=(mean_intra, arrow_y_pos), xytext=(mid_point, arrow_y_pos),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
# 右半边箭头：从中间指向 Inter
ax.annotate('', xy=(mean_inter, arrow_y_pos), xytext=(mid_point, arrow_y_pos),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# ================= 替换结束 =================

# 6. 设置标签和图例
ax.set_xlabel('Cosine Distance') # 修改为余弦距离
ax.set_ylabel('Frequency')       # DEEN 使用 Frequency
ax.legend(loc='upper right')

# 去掉上方和右方的边框，这也是论文常见的 clean style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("deen_style_distance.pdf", dpi=300, format='pdf') # 论文推荐用 PDF 或 SVG
plt.savefig("deen_style_distance.png", dpi=300)
plt.show()

print("Plot saved as deen_style_distance.png")
print(f"Intra mean: {mean_intra:.4f}")
print(f"Inter mean: {mean_inter:.4f}")
print(f"Delta: {d_val:.4f}")