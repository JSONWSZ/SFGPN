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

class Discriminator_ModalityClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator_ModalityClassifier, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.bn = nn.BatchNorm1d(output_size)
        self.f = f

    def forward(self, x):
        x = self.map1(x)
        x = self.map3(x)
        return self.bn(x)


# #####################################################################

class SoftmaxEntropyLoss(object):
    def __init__(self):
        self.nx = None
        self.ny = None
        self.loss = None
        # self.softmax = None
        # self.entropy = None
        # self.dnx = None

    def __call__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.loss = torch.mean(-torch.sum(ny * torch.log(F.softmax(nx, dim=1)), dim=1))
        return self.loss


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


# ========================== FRL: FocusModule ==========================

class FocusModule(nn.Module):
    """
    Foreground Region Localization (FRL) Module.
    Generates foreground heatmaps from layer2 features through
    upsampling and bidirectional strip spatial attention.

    Input:  (B, 512, H/8, W/8)  — layer2 output, 288x144 → 36x18
    Output: (B, 1, H, W)         — foreground heatmap at original resolution
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
        # Upsampling path
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)

        # Bidirectional strip spatial attention
        b, c, h, w = x.size()
        strip_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        strip_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        strip_h = self.conv1d(strip_h)
        strip_w = self.conv1d(strip_w)

        strip_h = self.sigmoid(self.norm(strip_h)).view(b, c, h, 1)
        strip_w = self.sigmoid(self.norm(strip_w)).view(b, c, 1, w)

        x = strip_h * strip_w * x

        # Output foreground heatmap
        x = self.out_conv(x)
        return x


# =====================================================================


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


class embed_net(nn.Module):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

        # ====== FRL modules: one per modality ======
        # layer2 output channels = 512, spatial size = H/8 × W/8 (36×18 for 288×144)
        self.vis_focus = FocusModule(in_channels=512)
        self.ir_focus = FocusModule(in_channels=512)

    def forward(self, x1, x2, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block: layer1
        x = self.base_resnet.base.layer1(x)

        # shared block: layer2 (with Non-local if enabled)
        if self.non_local == 'on':
            NL2_counter = 0
            if len(self.NL_2_idx) == 0:
                self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
        else:
            x = self.base_resnet.base.layer2(x)

        # ====== FRL: generate foreground heatmaps after layer2 ======
        if modal == 0:
            B = x.shape[0] // 2
            feat_vis = x[:B]
            feat_ir = x[B:]

            heatmap_vis = self.vis_focus(feat_vis)
            heatmap_ir = self.ir_focus(feat_ir)
            heatmaps = torch.cat((heatmap_vis, heatmap_ir), 0)
        else:
            # During inference (modal=1 or 2), no heatmaps needed
            heatmaps = torch.zeros(x.size(0), 1, x.size(2) * 8, x.size(3) * 8).to(x.device)

        # shared block: layer3 + Non-local
        if self.non_local == 'on':
            NL3_counter = 0
            if len(self.NL_3_idx) == 0:
                self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0:
                self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet.base.layer3(x)
            x = self.base_resnet.base.layer4(x)

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
            return x_pool, self.classifier(feat), feat, heatmaps
        else:
            return self.l2norm(x_pool), self.l2norm(feat)
