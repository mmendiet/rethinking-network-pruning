import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from models.slimmable_ops import USBatchNorm2d, USConv2d, USLinear, make_divisible
# from utils.config import FLAGS

from .channel_selection import channel_selection


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        ########## ADDED
        self.select = channel_selection(in_planes)
        ###########
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            # x = self.relu1(self.bn1(x))
            x = self.relu1(self.select(self.bn1(x)))
        else:
            # out = self.relu1(self.bn1(x))
            out = self.relu1(self.select(self.bn1(x)))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class Model(nn.Module):
    def __init__(self, num_classes=10, input_size=32, cfg=None):
        super(Model, self).__init__()
        depth, widen_factor, dropRate = 28, 10, 0.0
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False) #, us=[False, True])
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        ########  ADDED
        self.select = channel_selection(nChannels[3])
        ########
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes) #, us=[True, False])
        self.nChannels = nChannels[3]
        if cfg is not None:
            self.apply_cfg(cfg, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def apply_cfg(self, cfg, num_classes):
        layers = []
        for m in self.named_modules():
            if (isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d)
                or isinstance(m[1], nn.Linear) or isinstance(m[1], channel_selection)):
                layers.append(m)

        cfg_idx = 0
        for m in layers:
            if isinstance(m[1], nn.Conv2d):
                keys = m[0].split('.')
                if len(keys) > 1:
                    if 'conv1' in keys:
                        self._modules[keys[0]]._modules[keys[1]]._modules[keys[2]]._modules[keys[3]] = \
                            nn.Conv2d(cfg[cfg_idx-1], cfg[cfg_idx], kernel_size=m[1].kernel_size,
                                stride=m[1].stride, padding=m[1].padding, bias=m[1].bias)
                    elif 'conv2' in keys:
                            self._modules[keys[0]]._modules[keys[1]]._modules[keys[2]]._modules[keys[3]] = \
                            nn.Conv2d(cfg[cfg_idx-1], m[1].out_channels, kernel_size=m[1].kernel_size,
                                stride=m[1].stride, padding=m[1].padding, bias=m[1].bias)
                    elif 'convShortcut' in keys:
                        self._modules[keys[0]]._modules[keys[1]]._modules[keys[2]]._modules[keys[3]] = \
                            nn.Conv2d(cfg[cfg_idx-2], m[1].out_channels, kernel_size=m[1].kernel_size,
                                stride=m[1].stride, padding=m[1].padding, bias=m[1].bias)
            elif isinstance(m[1], nn.BatchNorm2d):
                keys = m[0].split('.')
                if len(keys) > 1 and 'bn1' not in keys:
                    self._modules[keys[0]]._modules[keys[1]]._modules[keys[2]]._modules[keys[3]] = nn.BatchNorm2d(cfg[cfg_idx]) 
                cfg_idx += 1
            elif isinstance(m[1], nn.Linear):
                self._modules[m[0]] = nn.Linear(cfg[-1], num_classes)
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        # out = self.relu(self.bn1(out))
        out = self.relu(self.select(self.bn1(out)))
        # out = F.avg_pool2d(out, 8)
        out = F.adaptive_avg_pool2d(out, output_size=(1, 1))
        last_dim = out.size()[1]
        out = out.view(-1, last_dim)
        return self.fc(out)