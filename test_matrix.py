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

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Test')
parser.add_argument('--dataset', default='sysu', help='dataset name: llcm, regdb or sysu]')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='sysu_agw_p4_n6_lr_0.1_seed_0_best.t', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--test-batch', default=32, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')  # SYSU-MM01

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../Datasets/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
    pool_dim = 2048
elif dataset == 'regdb':
    data_path = '../Datasets/RegDB/'
    n_class = 206
    test_mode = [2, 1]
    pool_dim = 1024
elif dataset == 'llcm':
    data_path = '../Datasets/LLCM/'
    n_class = 713
    test_mode = [2, 1]  # [1, 2]: IR to VIS; [2, 1]: VIS to IR;
    pool_dim = 2048

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0
print('==> Building model..')
net = embed_net(n_class, gm_pool='off', arch=args.arch)
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
    gall_feat1 = np.zeros((ngall, 2048))
    gall_feat_att1 = np.zeros((ngall, 2048))

    gall_feat2 = np.zeros((ngall, 2048))
    gall_feat_att2 = np.zeros((ngall, 2048))

    gall_feat3 = np.zeros((ngall, 2048))
    gall_feat_att3 = np.zeros((ngall, 2048))
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


if dataset == 'llcm':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
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

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2, query_feat3,query_feat_att1,query_feat_att2,query_feat_att3 = extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat1, gall_feat2, gall_feat3,gall_feat_att1,gall_feat_att2,gall_feat_att3 = extract_gall_feat(trial_gall_loader)

        # fc feature

        distmat1 = np.matmul(query_feat1, np.transpose(gall_feat1))
        distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))
        distmat3 = np.matmul(query_feat3, np.transpose(gall_feat3))
        distmat_att1 = np.matmul(query_feat_att1, np.transpose(gall_feat_att1))
        distmat_att2 = np.matmul(query_feat_att2, np.transpose(gall_feat_att2))
        distmat_att3 = np.matmul(query_feat_att3, np.transpose(gall_feat_att3))
        a = 0.3
        distmat = (distmat1 + distmat2 + distmat3)/3
        distmat_att=(distmat_att1+distmat_att2+distmat_att3)/3
        distmat_all = a * distmat + (1 - a) * distmat_att

        cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        cmc_all, mAP_all, mINP_all = eval_llcm(-distmat_all, query_label, gall_label, query_cam, gall_cam)


        if trial == 0:
            sum_cmc = cmc
            sum_mAP = mAP
            sum_mINP = mINP

            sum_cmc_att = cmc_att
            sum_mAP_att = mAP_att
            sum_mINP_att = mINP_att

            sum_cmc_all = cmc_all
            sum_mAP_all = mAP_all
            sum_mINP_all = mINP_all

        else:
            sum_cmc += cmc
            sum_mAP += mAP
            sum_mINP += mINP

            sum_cmc_att += cmc_att
            sum_mAP_att += mAP_att
            sum_mINP_att += mINP_att

            sum_cmc_all += cmc_all
            sum_mAP_all += mAP_all
            sum_mINP_all += mINP_all

        print('Test Trial: {}'.format(trial))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print(
            'ALL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))

elif dataset == 'sysu':

    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
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

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat1, query_feat2, query_feat3,query_feat_att1,query_feat_att2,query_feat_att3 = extract_query_feat(query_loader)
    for trial in range(10):
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial)

        trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)

        gall_feat1, gall_feat2, gall_feat3,gall_feat_att1,gall_feat_att2,gall_feat_att3 = extract_gall_feat(trial_gall_loader)


        a = 0.3
        distmat = (
                          np.matmul(query_feat1, np.transpose(gall_feat1)) +
                          np.matmul(query_feat2,np.transpose(gall_feat2)) +
                          np.matmul(query_feat3, np.transpose(gall_feat3))
                  ) / 3

        distmat_att = (
                              np.matmul(query_feat_att1, np.transpose(gall_feat_att1)) +
                              np.matmul(query_feat_att2,np.transpose(gall_feat_att2)) +
                              np.matmul(query_feat_att3, np.transpose(gall_feat_att3))
                      ) / 3

        distmat_all = a * distmat + (1 - a) * distmat_att

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        cmc_all, mAP_all, mINP_all = eval_sysu(-distmat_all, query_label, gall_label, query_cam, gall_cam)
        if trial == 0:
            sum_cmc = cmc
            sum_mAP = mAP
            sum_mINP = mINP

            sum_cmc_att = cmc_att
            sum_mAP_att = mAP_att
            sum_mINP_att = mINP_att

            sum_cmc_all = cmc_all
            sum_mAP_all = mAP_all
            sum_mINP_all = mINP_all

        else:
            sum_cmc += cmc
            sum_mAP += mAP
            sum_mINP += mINP

            sum_cmc_att += cmc_att
            sum_mAP_att += mAP_att
            sum_mINP_att += mINP_att

            sum_cmc_all += cmc_all
            sum_mAP_all += mAP_all
            sum_mINP_all += mINP_all

        print('Test Trial: {}'.format(trial))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print(
            'ALL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))

elif dataset == 'regdb':

    for trial in range(10):
        test_trial = trial + 1
        model_path = checkpoint_path + 'regdb_agw_p4_n6_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(model_path))
            checkpoint = torch.load(model_path,weights_only=False)
            net.load_state_dict(checkpoint['net'])

        # testing set
        query_img, query_label = process_test_regdb(data_path, trial=test_trial, modal='visible')
        gall_img, gall_label = process_test_regdb(data_path, trial=test_trial, modal='thermal')

        gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
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


        if trial == 0:
            sum_cmc = cmc
            sum_mAP = mAP
            sum_mINP = mINP

            sum_cmc_att = cmc_att
            sum_mAP_att = mAP_att
            sum_mINP_att = mINP_att

            sum_cmc_all = cmc_all
            sum_mAP_all = mAP_all
            sum_mINP_all = mINP_all

        else:
            sum_cmc += cmc
            sum_mAP += mAP
            sum_mINP += mINP

            sum_cmc_att += cmc_att
            sum_mAP_att += mAP_att
            sum_mINP_att += mINP_att

            sum_cmc_all += cmc_all
            sum_mAP_all += mAP_all
            sum_mINP_all += mINP_all

        print('Test Trial: {}'.format(trial))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print(
            'ALL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))

cmc = sum_cmc / 10
mAP = sum_mAP / 10
mINP = sum_mINP / 10

cmc_att = sum_cmc_att / 10
mAP_att = sum_mAP_att / 10
mINP_att = sum_mINP_att / 10

cmc_all = sum_cmc_all / 10
mAP_all = sum_mAP_all / 10
mINP_all = sum_mINP_all / 10

print('All Average:')
print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
print('ALL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
    cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))