# Copyright 2020 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.model_utils import get_activation

sig = nn.Sigmoid()
ACTIVATION = nn.ReLU
#device = 'cuda'


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.reshape(x.size()[0], -1)

def crop_and_concat(upsampled, bypass, crop=False):
    if crop:
        c = (bypass.size()[2] - upsampled.size()[2]) // 2
        bypass = F.pad(bypass, (-c, -c, -c, -c))

    return torch.cat((upsampled, bypass), 1)



def get_normalization(out_channels, method):
    if method == 'batch':
        return nn.BatchNorm2d(out_channels)
    elif method == 'instance':
        return nn.InstanceNorm2d(out_channels)
    elif method == 'group':
        return nn.GroupNorm(32, out_channels)
    elif method == 'none':
        return nn.Identity()

def get_normalization_3d(out_channels, method):
    if method == 'batch':
        return nn.BatchNorm3d(out_channels)
    elif method == 'instance':
        return nn.InstanceNorm3d(out_channels)
    elif method == 'group':
        return nn.GroupNorm(32, out_channels)
    elif method == 'none':
        return nn.Identity()


def conv2d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION, norm='batch'):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=1),
        #nn.BatchNorm2d(out_channels, momentum=momentum),
        get_normalization(out_channels, method=norm),
        activation(),
    )


def deconv2d_bn_block(in_channels, out_channels, use_upsample=True, kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION, norm='batch'):
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose2d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        #nn.BatchNorm2d(out_channels, momentum=momentum),
        get_normalization(out_channels, method=norm),
        activation(),
    )


def conv3d_bn_block(in_channels, out_channels, kernel=3, momentum=0.01, activation=ACTIVATION, norm='batch'):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel, padding=1),
        #nn.BatchNorm3d(out_channels, momentum=momentum),
        get_normalization_3d(out_channels, method=norm),
        activation(),
    )


def deconv3d_bn_block(in_channels, out_channels, use_upsample=(2, 2, 2), kernel=4, stride=2, padding=1, momentum=0.01,
                      activation=ACTIVATION, norm='batch'):
    if use_upsample:
        up = nn.Sequential(
            nn.Upsample(scale_factor=use_upsample),
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1)
        )
    else:
        up = nn.ConvTranspose3d(in_channels, out_channels, kernel, stride=stride, padding=padding)
    return nn.Sequential(
        up,
        #nn.BatchNorm3d(out_channels, momentum=momentum),
        get_normalization_3d(out_channels, method=norm),
        activation(),
    )


class Upsample(nn.Module):
    def __init__(self, num_channels, scale, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.scale = scale

        # torch.nn.utils.weight_norm
        self.conv0 = WeightNorm((nn.Conv2d(num_channels, scale, kernel_size, padding=padding)), ['weight'])

    # pixel shuffle
    def pixel_shuffle(self, x, scale):
        """https://gist.github.com/davidaknowles/6e95a643adaf3960d1648a6b369e9d0b"""
        num_batches, num_channels, nx, ny = x.shape
        num_channels = num_channels // scale
        out = x.contiguous().view(num_batches, num_channels, scale, nx, ny)
        out = out.permute(0, 1, 3, 2, 4).contiguous()
        out = out.view(num_batches, num_channels, nx * scale, ny)
        return out

    def forward(self, x):
        #print(x.shape)
        out = self.conv0(x)
        out = self.pixel_shuffle(out, self.scale)
        #print(out.shape)
        return out

#wn = lambda x: torch.nn.utils.weight_norm(x)


class Generator(nn.Module):
    def __init__(self, n_channels=1, out_channels=1, nf=32, norm_type='batch', activation=ACTIVATION, final='tanh', mc=False):
        super(Generator, self).__init__()

        #conv2_block = conv2d_bn_block
        conv3_block = conv3d_bn_block

        #max2_pool = nn.MaxPool2d(2)
        self.max3_pool = nn.MaxPool3d(2)
        act = activation

        if mc:
            dropout = 0.5
        else:
            dropout = 0.0

        self.down0 = nn.Sequential(
            conv3_block(n_channels, nf, activation=act, norm='none'),
            conv3_block(nf, nf, activation=act, norm=norm_type)
        )
        self.down1 = nn.Sequential(
            conv3_block(nf, 2 * nf, activation=act, norm=norm_type),
            conv3_block(2 * nf, 2 * nf, activation=act, norm=norm_type),
        )
        self.down2 = nn.Sequential(
            conv3_block(2 * nf, 4 * nf, activation=act, norm=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv3_block(4 * nf, 4 * nf, activation=act, norm=norm_type),

        )
        #self.down3 = nn.Sequential(
        #    conv3_block(4 * nf, 8 * nf, activation=act, norm=norm_type),
        #    nn.Dropout(p=dropout, inplace=False),
        #    conv3_block(8 * nf, 8 * nf, activation=act, norm=norm_type),
        #)

        #self.up3 = deconv3d_bn_block(8 * nf, 4 * nf, activation=act, norm=norm_type)

        self.conv5 = nn.Sequential(
            conv3_block(4 * nf, 4 * nf, activation=act, norm=norm_type),  # 8
            nn.Dropout(p=dropout, inplace=False),
            conv3_block(4 * nf, 4 * nf, activation=act, norm=norm_type),
        )
        self.up2 = deconv3d_bn_block(4 * nf, 2 * nf, activation=act, norm=norm_type)
        self.conv6 = nn.Sequential(
            conv3_block(4 * nf, 2 * nf, activation=act, norm=norm_type),
            nn.Dropout(p=dropout, inplace=False),
            conv3_block(2 * nf, 2 * nf, activation=act, norm=norm_type),
        )

        self.up1 = deconv3d_bn_block(2 * nf, nf, activation=act, norm=norm_type)

        final_layer = get_activation(final)

        self.conv7_k = nn.Sequential(
            conv3_block(2 * nf, out_channels, activation=final_layer, norm='none'),
        )

        self.conv7_g = nn.Sequential(
            conv3_block(2 * nf, out_channels, activation=final_layer, norm='none'),
        )

        self.encoder = nn.Sequential(self.down0, self.down1, self.down2, self.down3)
        self.decoder = nn.Sequential(self.up3, self.conv5, self.up2, self.conv6, self.up1)

        self.skip = Upsample(1, 8, 5)

    def forward(self, x, method=None):
        # x (1, C, X, Y, Z)
        if method != 'decode':
            # skip
            xdown = x[:, :, :, :, 4::8]
            skip = self.skip(xdown.permute(2, 1, 4, 3, 0).squeeze(4))  # (X, C, Z, Y)
            skip = skip.permute(1, 0, 3, 2).unsqueeze(0) #.detach()

            #x = x.permute(4, 1, 2, 3, 0)[:, :, :, :, 0]  # (Z, C, X, Y)
            feat = []
            for i in range(len(self.encoder)):
                #x = x.permute(1, 2, 3, 0).unsqueeze(0)  # (1, C, X, Y, Z)
                if i > 0:
                    x = self.max3_pool(x)
                #x = x.squeeze(0).permute(3, 0, 1, 2)  # (Z, C, X, Y)
                x = self.encoder[i](x)
                feat.append(x)#(x.permute(1, 2, 3, 0).unsqueeze(0))
            feat.append(skip)
            if method == 'encode':
                return feat

        alpha = 1
        if method == 'decode':
            feat = x

        [x0, x1, x2, skip] = feat

        x5 = self.conv5(x2)  # Dropout
        xu2 = self.up2(x5)
        # alpha
        x1 = alpha * x1 + (1 - alpha) * xu2
        xu2_ = xu2  # .detach()
        cat2 = torch.cat([xu2_, x1], 1)
        x6 = self.conv6(cat2)  # Dropout
        xu1 = self.up1(x6)
        xu1_ = xu1  # .detach()
        cat1 = torch.cat([xu1_, x0], 1)
        x70 = self.conv7_k(cat1)
        x71 = self.conv7_g(cat1)

        x70 = (x70 + skip) * 0.5#.detach()
        x71 = (x71 + skip) * 0.5#.detach()

        return {'out0': x70, 'out1': x71}


class WeightNorm(nn.Module):
    append_g = '_g'
    append_v = '_v'

    def __init__(self, module, weights):
        super(WeightNorm, self).__init__()
        self.module = module
        self.weights = weights
        self._reset()

    def _reset(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)

            # construct g,v such that w = g/||v|| * v
            with torch.no_grad():
                g = torch.norm(w)
                v = w/g.expand_as(w)
            g = nn.Parameter(g.data, requires_grad=True)
            v = nn.Parameter(v.data, requires_grad=True)
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v

            # remove w from parameter list
            del self.module._parameters[name_w]

            # add g and v as new parameters
            self.module.register_parameter(name_g, g)
            self.module.register_parameter(name_v, v)

    def _setweights(self):
        for name_w in self.weights:
            name_g = name_w + self.append_g
            name_v = name_w + self.append_v
            g = getattr(self.module, name_g)
            v = getattr(self.module, name_v)
            w = v*(g/torch.norm(v)).expand_as(v)
            setattr(self.module, name_w, w)

    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)


if __name__ == '__main__':
    g = Generator(n_channels=1, norm_type='group', final='tanh')
    #from torchsummary import summary
    #from utils.data_utils import print_num_of_parameters
    #print_num_of_parameters(g)
    f = g(torch.rand(1, 1, 128, 128, 128), method='encode')
    for ff in f:
        print(ff.shape)
    #f = g(torch.rand(1, 1, 128, 128, 16), method='encode')
    #print(f[-1].shape)
    #upsample = torch.nn.Upsample(size=(16, 16, 16))
    #fup = upsample(f[-1])
    #print(fup.shape)
    out = g(f, method='decode')
    print(out['out0'].shape)

    print(g(torch.rand(1, 1, 128, 128, 128))['out0'].shape)