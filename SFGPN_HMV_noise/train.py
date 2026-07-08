from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData_mask, RegDBData_mask, LLCMData_mask, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_llcm
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT, modality_center_loss
from tensorboardX import SummaryWriter
from random_erasing import RandomErasing
import math
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=10, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=384, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=6, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--erasing_p', default=0.5, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--num_pos', default=4, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--noise_kernel', default=0, type=int, help='mask noise kernel size: 0=clean, 5=L1, 11=L2, 21=L3')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../Datasets/SYSU-MM01-npy/'
    test_path = '../Datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
    pool_dim = 2048
elif dataset == 'regdb':
    data_path = '../Datasets/RegDB/'
    pedestrian_dir = '../Datasets/RegDB-Pedestrian'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
    pool_dim = 1024
elif dataset == 'llcm':
    data_path = '../Datasets/LLCM/'
    pedestrian_path = '../Datasets/LLCM-bag/'
    log_path = args.log_path + 'llcm_log/'
    test_mode = [1, 2]  # [2, 1]: VIS to IR; [1, 2]: IR to VIS
    pool_dim = 2048

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method == 'agw':
    suffix = suffix + '_agw_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

if args.noise_kernel > 0:
    suffix = suffix + '_hmv_noise_k{}'.format(args.noise_kernel)
else:
    suffix = suffix + '_hmv_clean'

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData_mask(data_path, transform_train=transform_train, where="body",
                             noise_kernel=args.noise_kernel)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(test_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(test_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData_mask(data_path, pedestrian_dir, args.trial, transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

elif dataset == 'llcm':
    # training set
    trainset = LLCMData_mask(data_path, pedestrian_path, transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_llcm(data_path, mode=test_mode[1])
    gall_img, gall_label, gall_cam = process_gallery_llcm(data_path, mode=test_mode[0], trial=0)

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
net = embed_net(n_class, gm_pool='off', arch=args.arch, dataset=dataset)
net.to(device)
cudnn.benchmark = True

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

# define loss function
criterion_id = nn.CrossEntropyLoss()
criterion_tri = TripletLoss_WRT()

criterion_id.to(device)
criterion_tri.to(device)
if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr},
    ],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 30:
        lr = 0.1
    elif epoch >= 30 and epoch < 65:
        lr = 0.01
    elif epoch >= 65 and epoch < 75:
        lr = 0.1
    elif epoch >= 75 and epoch < 115:
        lr = 0.01
    elif epoch >= 115:
        lr = 0.001

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    HC_loss = AverageMeter()
    twCompact_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct_cls = 0
    correct_tri = 0
    total = 0
    acc_cls = 0
    acc_tri = 0
    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (
            vis_pure_aug, vis_pedestrian_mask, vis_label, ir_pure_aug, ir_pedestrian_mask, ir_label) in enumerate(
        trainloader):
        # 合并标签
        labels = torch.cat((vis_label, ir_label, vis_label, ir_label, vis_label, ir_label), 0).long().cuda()
        lbs = torch.cat((vis_label, ir_label), 0).long().cuda()
        vis_pure_aug = vis_pure_aug.cuda()
        ir_pure_aug = ir_pure_aug.cuda()
        vis_mask = vis_pedestrian_mask.cuda()
        ir_mask = ir_pedestrian_mask.cuda()

        data_time.update(time.time() - end)

        # 前向传播 - HMV: 掩码传入模型参与硬乘法
        feat, out = net(vis_pure_aug, ir_pure_aug,
                        vis_mask=vis_mask, ir_mask=ir_mask)
        feat1, feat2, feat3 = torch.chunk(feat, 3, dim=0)

        loss_id = criterion_id(out, labels)

        loss_tri1, batch_acc1 = criterion_tri(feat1, lbs)
        loss_tri2, batch_acc2 = criterion_tri(feat2, lbs)
        loss_tri3, batch_acc3 = criterion_tri(feat3, lbs)
        loss_tri = (loss_tri1 + loss_tri2 + loss_tri3) / 3

        # 计算准确率
        _, predicted = out.max(1)
        correct_cls += predicted.eq(labels).sum().item()
        correct_tri += (batch_acc1 + batch_acc2 + batch_acc3)
        total += labels.size(0)

        # 总损失（无 SCG/FocalLoss）
        if epoch >= 65:
            loss_HC1, loss_twCompact1 = modality_center_loss(feat1, lbs)
            loss_HC2, loss_twCompact2 = modality_center_loss(feat2, lbs)
            loss_HC3, loss_twCompact3 = modality_center_loss(feat3, lbs)
            loss_HC = (loss_HC1 + loss_HC2 + loss_HC3) / 3
            loss_twCompact = (loss_twCompact1 + loss_twCompact2 + loss_twCompact3) / 3

            loss = loss_id + loss_tri + 1.7 * loss_HC + 0.24 * loss_twCompact
        else:
            loss = loss_id + loss_tri

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 更新统计量
        batch_size = vis_pure_aug.size(0)
        train_loss.update(loss.item(), batch_size)
        id_loss.update(loss_id.item(), batch_size)
        tri_loss.update(loss_tri.item(), batch_size)
        if epoch >= 65:
            HC_loss.update(loss_HC.item(), batch_size)
            twCompact_loss.update(loss_twCompact.item(), batch_size)

        # 计时
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % 50 == 0:
            acc_cls = 100. * correct_cls / total
            acc_tri = 100. * correct_tri / total
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr: {:.5f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'i: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'T: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'ClsAcc: {:.2f}% '
                  'TriAcc: {:.2f}%'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                acc_cls, acc_tri, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))
            if epoch >= 65:
                print('HC: {HC_loss.val:.4f} ({HC_loss.avg:.4f}) '
                      'twC: {twCompact_loss.val:.4f} ({twCompact_loss.avg:.4f}) '.format(
                          HC_loss=HC_loss, twCompact_loss=twCompact_loss))

    # 记录tensorboard
    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    writer.add_scalar('ClsAcc', acc_cls, epoch)
    writer.add_scalar('TriAcc', acc_tri, epoch)
    if epoch >= 65:
        writer.add_scalar('HC_loss', HC_loss.avg, epoch)
        writer.add_scalar('twCompact_loss', twCompact_loss.avg, epoch)


def test(epoch):
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

    start = time.time()
    # compute the similarity
    distmat = (np.matmul(query_feat1, np.transpose(gall_feat1)) + np.matmul(query_feat2,
                                                                            np.transpose(gall_feat2)) + np.matmul(
        query_feat3, np.transpose(gall_feat3))) / 3
    distmat_att = (np.matmul(query_feat_att1, np.transpose(gall_feat_att1)) + np.matmul(query_feat_att2, np.transpose(
        gall_feat_att2)) + np.matmul(query_feat_att3, np.transpose(gall_feat_att3))) / 3
    
    a=0.3
    distmat_all = a * distmat + (1-a) * distmat_att

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
        cmc_all, mAP_all, mINP_all = eval_regdb(-distmat_all, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        cmc_all, mAP_all, mINP_all = eval_sysu(-distmat_all, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'llcm':
        cmc, mAP, mINP = eval_llcm(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_llcm(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        cmc_all, mAP_all, mINP_all = eval_llcm(-distmat_all, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    writer.add_scalar('rank1_all', cmc_all[0], epoch)
    writer.add_scalar('mAP_all', mAP_all, epoch)
    writer.add_scalar('mINP_all', mINP_all, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, cmc_all, mAP_all, mINP_all


# training
print('==> Start Training...')
for epoch in range(start_epoch, 151):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label,
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch,
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    print('Test Epoch: {}'.format(epoch))

    # testing
    cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, cmc_all, mAP_all, mINP_all = test(epoch)
    # save model
    if cmc_all[0] > best_acc:  # not the real best for sysu-mm01
        best_acc = cmc_all[0]
        best_epoch = epoch
        state = {
            'net': net.state_dict(),
            'cmc': cmc_all,
            'mAP': mAP_all,
            'mINP': mINP_all,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_best.t')

    if epoch == 150:
        state = {
            'net': net.state_dict(),
            'cmc': cmc_all,
            'mAP': mAP_all,
            'mINP': mINP_all,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

    print(
        'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print(
        'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
    print(
        'all:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))
    print('Best Epoch [{}]'.format(best_epoch))
