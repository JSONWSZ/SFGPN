from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
import shutil
from PIL import Image


parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Test')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--test-batch', default=32, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

data_path = '../Datasets/RegDB/'
n_class = 206
test_mode = [1, 2]
pool_dim = 1024


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
print('==> Building model..')
net = embed_net(n_class, gm_pool='off', arch=args.arch, dataset='regdb')
net.to(device)
cudnn.benchmark = True

checkpoint_path = args.model_path


print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()


def extract_gall_feat(gall_loader):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat1 = np.zeros((ngall, pool_dim))
    gall_feat_att1 = np.zeros((ngall, pool_dim))

    gall_feat2 = np.zeros((ngall, pool_dim))
    gall_feat_att2 = np.zeros((ngall, pool_dim))

    gall_feat3 = np.zeros((ngall, pool_dim))
    gall_feat_att3 = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, modal=test_mode[0])

            gall_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            gall_feat_att1[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()

            gall_feat2[ptr:ptr + batch_num, :] = feat[batch_num:2 * batch_num].detach().cpu().numpy()
            gall_feat_att2[ptr:ptr + batch_num, :] = feat_att[batch_num:2 * batch_num].detach().cpu().numpy()

            gall_feat3[ptr:ptr + batch_num, :] = feat[2 * batch_num:3 * batch_num].detach().cpu().numpy()
            gall_feat_att3[ptr:ptr + batch_num, :] = feat_att[2 * batch_num:3 * batch_num].detach().cpu().numpy()

            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat1, gall_feat2, gall_feat3,gall_feat_att1,gall_feat_att2,gall_feat_att3


def extract_query_feat(query_loader):
    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat1 = np.zeros((nquery, pool_dim))
    query_feat_att1 = np.zeros((nquery, pool_dim))

    query_feat2 = np.zeros((nquery, pool_dim))
    query_feat_att2 = np.zeros((nquery, pool_dim))

    query_feat3 = np.zeros((nquery, pool_dim))
    query_feat_att3 = np.zeros((nquery, pool_dim))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, modal=test_mode[1])

            query_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            query_feat_att1[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()

            query_feat2[ptr:ptr + batch_num, :] = feat[batch_num:2 * batch_num].detach().cpu().numpy()
            query_feat_att2[ptr:ptr + batch_num, :] = feat_att[batch_num:2 * batch_num].detach().cpu().numpy()

            query_feat3[ptr:ptr + batch_num, :] = feat[2 * batch_num:3 * batch_num].detach().cpu().numpy()
            query_feat_att3[ptr:ptr + batch_num, :] = feat_att[2 * batch_num:3 * batch_num].detach().cpu().numpy()

            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat1, query_feat2, query_feat3,query_feat_att1,query_feat_att2,query_feat_att3


def copy_image(src_path, dst_path):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy(src_path, dst_path)


test_trial =  1
model_path = checkpoint_path + 'regdb_agw_p4_n6_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
if os.path.isfile(model_path):
    print('==> loading checkpoint {}'.format(model_path))
    checkpoint = torch.load(model_path,weights_only=False)
    net.load_state_dict(checkpoint['net'])

# testing set
query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')
gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='visible')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

nquery = len(query_label)
ngall = len(gall_label)

queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

query_feat1, query_feat2, query_feat3, query_feat_att1, query_feat_att2, query_feat_att3 = extract_query_feat(query_loader)
gall_feat1, gall_feat2, gall_feat3, gall_feat_att1, gall_feat_att2, gall_feat_att3 = extract_gall_feat(gall_loader)


distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
distmat_att1 = np.matmul(query_feat_att1, np.transpose(gall_feat_att1))
distmat_att2 = np.matmul(query_feat_att2, np.transpose(gall_feat_att2))
distmat_att3 = np.matmul(query_feat_att3, np.transpose(gall_feat_att3))
a = 0.3
distmat = (distmat1 + distmat2 + distmat3)/3
distmat_att = (distmat_att1+distmat_att2+distmat_att3)/3
distmat_all=a*distmat+(1-a)*distmat_att

cmc, mAP, mINP = eval_regdb(-distmat, gall_label, query_label)
cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, gall_label, query_label)
cmc_all, mAP_all, mINP_all = eval_regdb(-distmat_all, gall_label, query_label)




print('Test Trial: {}'.format(test_trial))
print(
    'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print(
    'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
print(
    'ALL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))


# =========================
# RegDB Retrieval Visualization
# =========================

save_root = r"E:\python\projects\Datasets\RegDB-ir2vis-trial1"
os.makedirs(save_root, exist_ok=True)

# 使用最终用于评估的距离矩阵
distmat_vis = distmat_all  # shape: [nquery, ngall]

query_count = 0  # 用于子文件夹编号（1,2,3,...）

print("==> Start visualizing retrieval results...")

for q_idx in range(nquery):

    q_pid = query_label[q_idx]
    q_img_path = query_img[q_idx]
    q_img_name = os.path.basename(q_img_path)

    # 对 gallery 排序（相似度从高到低）
    sorted_indices = np.argsort(-distmat_vis[q_idx])

    # Rank-1 gallery
    rank1_idx = sorted_indices[0]
    rank1_pid = gall_label[rank1_idx]

    # 只保存 Rank-1 不匹配的样例
    if rank1_pid == q_pid:
        continue

    # 创建子文件夹
    query_count += 1
    query_dir = os.path.join(save_root, str(query_count))
    os.makedirs(query_dir, exist_ok=True)

    # =====================
    # 0. 保存 Query 图像
    # =====================
    query_save_name = f"0_{q_img_name}"
    query_save_path = os.path.join(query_dir, query_save_name)
    copy_image(q_img_path, query_save_path)

    # =====================
    # 1~10 保存 Gallery 图像
    # =====================
    for rank in range(10):
        g_idx = sorted_indices[rank]
        g_pid = gall_label[g_idx]
        g_img_path = gall_img[g_idx]
        g_img_name = os.path.basename(g_img_path)

        is_match = (g_pid == q_pid)
        match_str = "True" if is_match else "False"

        save_name = f"{rank+1}_{g_img_name}_{match_str}.bmp"
        save_path = os.path.join(query_dir, save_name)

        copy_image(g_img_path, save_path)

    print(f"[Saved] Query {q_idx} -> Folder {query_count}")

print(f"==> Visualization finished. Total saved queries: {query_count}")
