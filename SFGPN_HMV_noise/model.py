import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torch.nn.functional as F


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio // reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class FocusModule(nn.Module):
    """
    FocusModule
    功能：通过上采样与双向条状空间注意力生成聚焦热力图，
    引导网络显式聚焦于行人身体区域。
    """

    def __init__(self, in_channels=512):
        super(FocusModule, self).__init__()

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.kernel_size = 7
        self.groups = 64
        self.num_groups = 16
        self.pad = self.kernel_size // 2
        self.conv1d = nn.Conv1d(
            64, 64, kernel_size=self.kernel_size,
            padding=self.pad, groups=self.groups, bias=False
        )
        self.norm = nn.GroupNorm(self.num_groups, 64)
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 上采样路径
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        # 双向空间注意力
        b, c, h, w = x.size()
        strip_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        strip_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        strip_h = self.conv1d(strip_h)
        strip_w = self.conv1d(strip_w)

        strip_h = self.sigmoid(self.norm(strip_h)).view(b, c, h, 1)
        strip_w = self.sigmoid(self.norm(strip_w)).view(b, c, 1, w)

        x = strip_h * strip_w * x

        # 输出聚焦热力图
        x = self.out_conv(x)
        return x


class MultiScaleModule(nn.Module):

    def __init__(self, channel, dilations=(1, 2, 3), pool_sizes=(1, 2, 3, 6)):
        super(MultiScaleModule, self).__init__()

        # ===== 分支1：空洞卷积 =====
        self.dconvs = nn.ModuleList([
            nn.Conv2d(channel, channel // 4, kernel_size=3, stride=1,
                      padding=d, dilation=d, bias=False)
            for d in dilations
        ])
        for m in self.dconvs:
            weights_init_kaiming(m)
        self.conv1x1_d = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.conv1x1_d.apply(weights_init_kaiming)

        # ===== 分支2：金字塔池化 =====
        self.pool_sizes = pool_sizes
        self.ppm_convs = nn.ModuleList([
            nn.Conv2d(channel, channel // 4, kernel_size=1, bias=False)
            for _ in pool_sizes
        ])
        for m in self.ppm_convs:
            weights_init_kaiming(m)
        self.conv1x1_ppm = nn.Conv2d(channel // 4, channel, kernel_size=1)
        self.conv1x1_ppm.apply(weights_init_kaiming)

    def forward(self, x):
        B, C, H, W = x.shape

        # ===== 分支1：空洞卷积 =====
        d_out = 0
        for conv in self.dconvs:
            d_out += conv(x)
        d_out = d_out / len(self.dconvs)  # 平均
        d_out = F.relu(d_out)
        d_out = self.conv1x1_d(d_out)  # 恢复到 C 通道

        # ===== 分支2：金字塔池化 =====
        ppm_feats = []
        for i, s in enumerate(self.pool_sizes):
            pooled = F.adaptive_avg_pool2d(x, output_size=(s, s))  # (B, C, s, s)
            convp = self.ppm_convs[i](pooled)  # (B, C//4, s, s)
            up = F.interpolate(convp, size=(H, W), mode='bilinear', align_corners=False)
            ppm_feats.append(up)
        ppm_out = torch.stack(ppm_feats, dim=0).mean(dim=0)  # 平均各尺度
        ppm_out = F.relu(ppm_out)
        ppm_out = self.conv1x1_ppm(ppm_out)  # 恢复到 C 通道

        # ===== 拼接输出 =====
        out = torch.cat([x, d_out, ppm_out], dim=0)  # (3*B, C, H, W)
        return out


class embed_net(nn.Module):
    """
    HMV (Hard Mask Variant): 去除 FRL 模块与 SCG 损失，
    在 ResNet layer2 之后将二值掩码逐元素乘到特征图上，
    实现硬掩码前景过滤。
    """
    def __init__(self, class_num, gm_pool='on', arch='resnet50', dataset='sysu'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        self.dataset = dataset
        if self.dataset == 'regdb':
            pool_dim = 1024
            self.multiscale = MultiScaleModule(512)
        else:
            pool_dim = 2048
            self.multiscale = MultiScaleModule(1024)

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

    def forward(self, vis, ir, vis_mask=None, ir_mask=None, modal=0):
        """
        Args:
            vis: 可见光图像 (B, 3, 384, 144)
            ir:  红外图像 (B, 3, 384, 144)
            vis_mask: 可见光二值掩码 (B, 1, 384, 144), 值为 0 或 1
            ir_mask:  红外二值掩码 (B, 1, 384, 144), 值为 0 或 1
            modal: 0=train, 1=VIS gallery, 2=IR query
        """
        if modal == 0:
            x1 = self.visible_module(vis)    # (B, 64, 96, 36)
            x2 = self.thermal_module(ir)     # (B, 64, 96, 36)
            x = torch.cat((x1, x2), 0)      # (2B, 64, 96, 36)

        elif modal == 1:
            x = self.visible_module(vis)
        elif modal == 2:
            x = self.thermal_module(ir)

        # 共享层 layer1
        x = self.base_resnet.base.layer1(x)  # (2B, 256, 96, 36)
        # 共享层 layer2
        x = self.base_resnet.base.layer2(x)  # (2B, 512, 48, 18)

        if modal == 0:
            B = x.shape[0] // 2
            feat_vis = x[:B]   # (B, 512, 48, 18)
            feat_ir = x[B:]    # (B, 512, 48, 18)

            # 掩码下采样到 layer2 输出的空间尺寸 (48, 18)
            mask_vis_down = F.interpolate(
                vis_mask, size=feat_vis.shape[-2:], mode='nearest'
            )  # (B, 1, 48, 18)
            mask_ir_down = F.interpolate(
                ir_mask, size=feat_ir.shape[-2:], mode='nearest'
            )  # (B, 1, 48, 18)

            # 硬掩码乘法：掩码逐元素乘到特征图的每个通道上
            feat_vis = feat_vis * mask_vis_down  # (B, 512, 48, 18)
            feat_ir  = feat_ir  * mask_ir_down   # (B, 512, 48, 18)

            x = torch.cat((feat_vis, feat_ir), 0)  # (2B, 512, 48, 18)

        if self.dataset == 'regdb':
            x = self.multiscale(x)
            x = self.base_resnet.base.layer3(x)
        else:
            # SYSU/LLCM: layer3 → CSFP → layer4
            x = self.base_resnet.base.layer3(x)  # (2B, 1024, 24, 9)
            x = self.multiscale(x)               # (6B, 1024, 24, 9)
            x = self.base_resnet.base.layer4(x)  # (6B, 2048, 24, 9)

        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x ** p, dim=-1) + 1e-12) ** (1 / p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)
