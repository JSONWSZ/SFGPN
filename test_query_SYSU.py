from __future__ import print_function
import argparse
import time
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData_mask, RegDBData, LLCMData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
import shutil
from PIL import Image
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='HC04_twCompact002-65_150.t', type=str, help='resume from checkpoint')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--trial', default=0, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

test_path = '../Datasets/SYSU-MM01/'
test_mode = [1, 2]  # thermal to visible

checkpoint_path = args.model_path

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# testing set
query_img, query_label, query_cam = process_query_sysu(test_path, mode=args.mode)
gall_img, gall_label, gall_cam = process_gallery_sysu(test_path, mode=args.mode, trial=args.trial)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

nquery = len(query_label)
ngall = len(gall_label)

print("Dataset statistics:")
print("  ------------------------------")
print("  subset   | # ids | # images")
print("  ------------------------------")
print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
print("  ------------------------------")
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(395, gm_pool='off', arch=args.arch)
net.to(device)
cudnn.benchmark = True

model_path = checkpoint_path + args.resume
if os.path.isfile(model_path):
    print('==> loading checkpoint {}'.format(args.resume))
    checkpoint = torch.load(model_path, weights_only=False)
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['net'])
    print('==> loaded checkpoint {} (epoch {})'
          .format(args.resume, checkpoint['epoch']))
else:
    print('==> no checkpoint found at {}'.format(args.resume))


def visualize_results(distmat_all, query_img, query_label, query_cam, gall_img, gall_label, gall_cam, trial, save_root):
    """
    可视化每个 query 的检索结果（仅当 Rank-1 不是正样本时才保存）
    """
    import os
    import shutil
    from PIL import Image
    import numpy as np

    save_dir = os.path.join(save_root, f"trial{trial}")
    os.makedirs(save_dir, exist_ok=True)

    n_query = len(query_img)
    n_gallery = len(gall_img)

    print(f"==> 开始保存可视化结果到 {save_dir}")

    saved_count = 0  # 统计保存的 query 数量

    for q_idx in range(n_query):
        q_label = query_label[q_idx]
        q_cam = query_cam[q_idx]
        q_path = query_img[q_idx]

        # 当前 query 与所有 gallery 的距离
        distances = distmat_all[q_idx]
        indices = np.argsort(distances)  # 从小到大排序

        # Rank-1 gallery
        top1_idx = indices[0]
        top1_label = gall_label[top1_idx]
        top1_cam = gall_cam[top1_idx]
        top1_path = gall_img[top1_idx]

        # 仅在 Rank-1 错误时保存
        # if top1_label == q_label:
        #     continue

        saved_count += 1
        query_folder = os.path.join(save_dir, f"{q_idx+1}")
        os.makedirs(query_folder, exist_ok=True)

        # 保存 query 图片
        try:
            q_img = Image.open(q_path).convert('RGB')
            q_save_name = f"0_id{q_label}_cam{q_cam}.jpg"
            q_img.save(os.path.join(query_folder, q_save_name))
        except Exception as e:
            print(f"[Warning] 无法保存 query 图像: {q_path}, 错误: {e}")
            continue

        # 记录正样本排名
        positive_ranks = []

        # 保存所有 gallery 图片的可视化（可以改成保存前 N 张）
        # top_k = min(50, n_gallery)  # 例如仅保存前 50 张
        top_k = min(10, n_gallery)  # 例如仅保存前 50 张
        for rank, idx in enumerate(indices[:top_k]):
            g_label = gall_label[idx]
            g_cam = gall_cam[idx]
            g_path = gall_img[idx]
            is_positive = (q_label == g_label)

            if is_positive:
                positive_ranks.append(rank + 1)

            save_name = f"{rank+1}_id{g_label}_cam{g_cam}_{is_positive}.jpg"
            save_path = os.path.join(query_folder, save_name)

            try:
                shutil.copy(g_path, save_path)
            except Exception as e:
                print(f"[Warning] 拷贝失败: {g_path} -> {save_path}, 错误: {e}")

        # 保存正样本排名信息
        results_txt = os.path.join(query_folder, "results.txt")
        with open(results_txt, "w", encoding="utf-8") as f:
            f.write(",".join(str(x) for x in positive_ranks))

    print(f"==> 可视化完成，仅保存了 Rank-1 错误的样本，共 {saved_count} 个 query。")


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat1 = np.zeros((ngall, 2048))
    gall_feat_att1= np.zeros((ngall, 2048))

    gall_feat2 = np.zeros((ngall, 2048))
    gall_feat_att2= np.zeros((ngall, 2048))

    gall_feat3 = np.zeros((ngall, 2048))
    gall_feat_att3= np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, modal=test_mode[0])

            gall_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            gall_feat_att1[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()

            gall_feat2[ptr:ptr + batch_num, :] = feat[batch_num:2*batch_num].detach().cpu().numpy()
            gall_feat_att2[ptr:ptr + batch_num, :] = feat_att[batch_num:2*batch_num].detach().cpu().numpy()

            gall_feat3[ptr:ptr + batch_num, :] = feat[2*batch_num:3*batch_num].detach().cpu().numpy()
            gall_feat_att3[ptr:ptr + batch_num, :] = feat_att[2*batch_num:3*batch_num].detach().cpu().numpy()


            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat1 = np.zeros((nquery, 2048))
    query_feat_att1 = np.zeros((nquery, 2048))

    query_feat2 = np.zeros((nquery, 2048))
    query_feat_att2 = np.zeros((nquery, 2048))

    query_feat3 = np.zeros((nquery, 2048))
    query_feat_att3 = np.zeros((nquery, 2048))

    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input,modal= test_mode[1])

            query_feat1[ptr:ptr + batch_num, :] = feat[:batch_num].detach().cpu().numpy()
            query_feat_att1[ptr:ptr + batch_num, :] = feat_att[:batch_num].detach().cpu().numpy()

            query_feat2[ptr:ptr + batch_num, :] = feat[batch_num:2*batch_num].detach().cpu().numpy()
            query_feat_att2[ptr:ptr + batch_num, :] = feat_att[batch_num:2*batch_num].detach().cpu().numpy()

            query_feat3[ptr:ptr + batch_num, :] = feat[2*batch_num:3*batch_num].detach().cpu().numpy()
            query_feat_att3[ptr:ptr + batch_num, :] = feat_att[2*batch_num:3*batch_num].detach().cpu().numpy()

            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = (np.matmul(query_feat1, np.transpose(gall_feat1)) + np.matmul(query_feat2, np.transpose(gall_feat2)) +np.matmul(query_feat3, np.transpose(gall_feat3)) )/3
    distmat_att =( np.matmul(query_feat_att1, np.transpose(gall_feat_att1)) + np.matmul(query_feat_att2, np.transpose(gall_feat_att2)) + np.matmul(query_feat_att3, np.transpose(gall_feat_att3)) )/3
    distmat_all =( distmat + distmat_att )/2


    cmc_all, mAP_all, mINP_all = eval_sysu(-distmat_all, query_label, gall_label, query_cam, gall_cam)

    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    save_root = r"E:\python\projects\Datasets\SYSU-MM01-trial"
    trial_dir = os.path.join(save_root, f"trial{args.trial}")
    os.makedirs(trial_dir, exist_ok=True)

    trial_txt = os.path.join(trial_dir, f"trial{args.trial}.txt")
    with open(trial_txt, "w", encoding="utf-8") as f:
        f.write(
            'ALL:    Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}\n'.format(
                cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))
    print(f"==> 测试指标结果已保存到: {trial_txt}")
    visualize_results(-distmat_all, query_img, query_label, query_cam, gall_img, gall_label, gall_cam, args.trial,
                      save_root)


test(0)

