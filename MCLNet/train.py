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
#import torchvision.transforms as transforms

import transforms

from data_loader import SYSUData, SYSUData_mask, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT, FocalLoss
from tensorboardX import SummaryWriter

from model import Discriminator_ModalityClassifier
from model import SoftmaxEntropyLoss

from MCCA_id_loss import *
from MCCA_cam_loss import *

# ====== Data paths — adjust to your environment ======
SYSU_DATA_PATH = "../Datasets/SYSU-MM01-npy/"
SYSU_TEST_PATH = "../Datasets/SYSU-MM01/"
REGDB_DATA_PATH = "REGDB_DATA_PATH"

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='mcl', type=str,
                    metavar='m', help='method type: base or mcl')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--p', default=0.8, type=float, help='Random Erasing probability')
parser.add_argument('--sh', default=0.8, type=float, help='max erasing area')
parser.add_argument('--r1', default=0.3, type=float, help='aspect of erasing area')
parser.add_argument('--sl', default=0.1, type=float, help='min erasing area')

parser.add_argument('--sigma', default=8, type=float, help='mcca center margin')

# ====== FRL + SCG options ======
parser.add_argument('--use_frl_scg', action='store_true', default=False,
                    help='Enable FRL module + SCG loss for plug-and-play experiment')
parser.add_argument('--scg_weight', default=0.001, type=float,
                    help='weight for SCG loss (default: 0.001)')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = SYSU_DATA_PATH
    test_path = SYSU_TEST_PATH
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = REGDB_DATA_PATH
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method == 'mcl':
    suffix = suffix + '_mcl_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

if args.use_frl_scg:
    suffix = suffix + '_FRL_SCG_w{}'.format(args.scg_weight)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

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

# Transform for training: only ToTensor + Normalize
# (mask-aware augmentations are done inside SYSUData_mask)
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
    # training set — use mask-aware version when FRL+SCG is enabled
    if args.use_frl_scg:
        trainset = SYSUData_mask(data_path, transform=transform_train)
        print('==> Using SYSUData_mask (with FRL+SCG)')
    else:
        trainset = SYSUData(data_path, transform=transform_train)
        print('==> Using SYSUData (original, no masks)')

    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(test_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(test_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

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
if args.method == 'base':
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)
else:
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
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

criterion_mccaid = MCCALossID(num_classes=395, feat_dim=2048, rho=4500, sigma=args.sigma, use_gpu=True)
criterion_mccacam = MCCALossCAM(num_classes=6, feat_dim=2048, rho=4500, sigma=args.sigma, use_gpu=True)

discriminator_activation_function = nn.Sigmoid()
D = Discriminator_ModalityClassifier(input_size=2048,
                                     hidden_size=128,
                                     output_size=2,
                                     f=discriminator_activation_function)
D.to(device)

if args.method == 'mcl':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)

criterion_id.to(device)
criterion_tri.to(device)

criterion_mccaid.to(device)
criterion_mccacam.to(device)

id_criterion = SoftmaxEntropyLoss()

# ====== SCG loss ======
if args.use_frl_scg:
    criterion_scg = FocalLoss()
    criterion_scg.to(device)
    print('==> SCG loss weight: {}'.format(args.scg_weight))


if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

d_optimizer = optim.SGD(D.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9, nesterov=True)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    modal_loss = AverageMeter()
    mcca_id_loss = AverageMeter()
    mcca_cam_loss = AverageMeter()
    scg_loss = AverageMeter()

    data_time = AverageMeter()
    batch_time = AverageMeter()

    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, batch_data in enumerate(trainloader):

        if args.use_frl_scg:
            # SYSUData_mask returns: img1, mask1, label1, camid1, img2, mask2, label2, camid2
            input1, mask1, label1, camid1, input2, mask2, label2, camid2 = batch_data
        else:
            # Original SYSUData returns: img1, img2, label1, label2, camid1, camid2
            input1, input2, label1, label2, camid1, camid2 = batch_data

        labels = torch.cat((label1, label2), 0)
        camids = torch.cat((camid1, camid2), 0)

        true_modals = torch.tensor([[1.0, 0.0]] * len(label1) + [[0.0, 1.0]] * len(label2))
        fake_modals = torch.tensor([[0.5, 0.5]] * len(label1) * 2)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        camids = Variable(camids.cuda())

        true_modals = Variable(true_modals.cuda())
        fake_modals = Variable(fake_modals.cuda())

        data_time.update(time.time() - end)

        # Forward pass — returns 4 values during training (with heatmaps)
        feat, out0, feat_norm, heatmaps = net(input1, input2)

        # 1. Train D on real+fake (detach feat to avoid retaining generator graph)
        d_optimizer.zero_grad()
        d_decision_d = D(feat.detach())
        d_loss = id_criterion(d_decision_d, true_modals)
        d_loss.backward()
        d_optimizer.step()

        # 2. Generator modality confusion loss (gradients flow through D back to generator)
        d_decision = D(feat)
        loss_modal = id_criterion(d_decision, fake_modals)
        loss_mcca_id = criterion_mccaid(feat_norm, labels)
        loss_mcca_cam = criterion_mccacam(feat_norm, camids)

        loss_id = criterion_id(out0, labels)
        loss_tri, batch_acc = criterion_tri(feat, labels)
        correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        # ====== SCG loss ======
        loss = loss_id + loss_tri + loss_modal + 0.0005 * loss_mcca_id + 0.0005 * loss_mcca_cam

        if args.use_frl_scg:
            masks = torch.cat((mask1, mask2), 0).cuda()
            loss_scg = criterion_scg(masks, heatmaps)
            loss = loss + args.scg_weight * loss_scg
            scg_loss.update(loss_scg.item(), 2 * input1.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        modal_loss.update(loss_modal.item(), 2 * input1.size(0))
        mcca_id_loss.update(loss_mcca_id.item(), 2 * input1.size(0))
        mcca_cam_loss.update(loss_mcca_cam.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            if args.use_frl_scg:
                print('Epoch: [{}][{}/{}] '
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'lr:{:.3f} '
                      'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                      'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                      'MLoss: {modal_loss.val:.4f} ({modal_loss.avg:.4f})'
                      'MCCAIDloss:{mcca_id_loss.val:.4f} ({mcca_id_loss.avg:.4f})'
                      'MCCACAMloss:{mcca_cam_loss.val:.4f} ({mcca_cam_loss.avg:.4f})'
                      'SCGloss:{scg_loss.val:.4f} ({scg_loss.avg:.4f})'
                      'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,
                    modal_loss=modal_loss, mcca_id_loss=mcca_id_loss,
                    mcca_cam_loss=mcca_cam_loss, scg_loss=scg_loss))
            else:
                print('Epoch: [{}][{}/{}] '
                      'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                      'lr:{:.3f} '
                      'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                      'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                      'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                      'MLoss: {modal_loss.val:.4f} ({modal_loss.avg:.4f})'
                      'MCCAIDloss:{mcca_id_loss.val:.4f} ({mcca_id_loss.avg:.4f})'
                      'MCCACAMloss:{mcca_cam_loss.val:.4f} ({mcca_cam_loss.avg:.4f})'
                      'Accu: {:.2f}'.format(
                    epoch, batch_idx, len(trainloader), current_lr,
                    100. * correct / total, batch_time=batch_time,
                    train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,
                    modal_loss=modal_loss, mcca_id_loss=mcca_id_loss,
                    mcca_cam_loss=mcca_cam_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('modal_loss', modal_loss.avg, epoch)
    writer.add_scalar('mcca_id_loss', mcca_id_loss.avg, epoch)
    writer.add_scalar('mcca_cam_loss', mcca_cam_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    if args.use_frl_scg:
        writer.add_scalar('scg_loss', scg_loss.avg, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))
    # Combined distance matrix (same as SFGPN: weighted sum of pool and fc)
    a = 0.3
    distmat_all = a * distmat + (1 - a) * distmat_att

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
        cmc_all, mAP_all, mINP_all  = eval_regdb(-distmat_all, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
        cmc_all, mAP_all, mINP_all = eval_sysu(-distmat_all, query_label, gall_label, query_cam, gall_cam)
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
for epoch in range(start_epoch, 201 - start_epoch):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label,
                              trainset.train_thermal_label, color_pos, thermal_pos,
                              args.num_pos, args.batch_size, epoch)

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

    # testing every epoch
    cmc, mAP, mINP, cmc_att, mAP_att, mINP_att, cmc_all, mAP_all, mINP_all = test(epoch)
    # save only the best model
    if cmc_all[0] > best_acc:
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

    print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
    print('ALL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_all[0], cmc_all[4], cmc_all[9], cmc_all[19], mAP_all, mINP_all))
    print('Best Epoch [{}]'.format(best_epoch))
