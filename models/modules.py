import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GroupNormConv2d', 'Inception_GroupNorm', 'CRelu_GroupNorm', 'RFE']

class DeformConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(DeformConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class GroupNormConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(GroupNormConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.gn = nn.GroupNorm(out_channels // 4, out_channels, eps=1e-5)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        return F.relu(x, inplace=True)

class Inception_GroupNorm(nn.Module):

  def __init__(self):
    super(Inception_GroupNorm, self).__init__(in_channels=128)
    self.branch1x1 = GroupNormConv2d(in_channels, 64, kernel_size=1, padding=0)
    self.branch1x1_2 = GroupNormConv2d(in_channels, 64, kernel_size=1, padding=0)
    self.branch3x3_reduce = GroupNormConv2d(in_channels, 24, kernel_size=1, padding=0)
    self.branch3x3 = GroupNormConv2d(24, 64, kernel_size=3, padding=1)
    self.branch3x3_reduce_2 = GroupNormConv2d(in_channels, 24, kernel_size=1, padding=0)
    self.branch3x3_2 = GroupNormConv2d(24, 32, kernel_size=3, padding=1)
    self.branch3x3_3 = GroupNormConv2d(32, 64, kernel_size=3, padding=1)

  def forward(self, x):
    branch1x1 = self.branch1x1(x)

    branch1x1_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch1x1_2 = self.branch1x1_2(branch1x1_pool)

    branch3x3_reduce = self.branch3x3_reduce(x)
    branch3x3 = self.branch3x3(branch3x3_reduce)

    branch3x3_reduce_2 = self.branch3x3_reduce_2(x)
    branch3x3_2 = self.branch3x3_2(branch3x3_reduce_2)
    branch3x3_3 = self.branch3x3_3(branch3x3_2)

    outputs = [branch1x1, branch1x1_2, branch3x3, branch3x3_3]
    return torch.cat(outputs, 1)

class CRelu_GroupNorm(nn.Module):

  def __init__(self, in_channels, out_channels, **kwargs):
    super(CRelu_GroupNorm, self).__init__()
    self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
    self.gn = nn.GroupNorm(out_channels // 4, out_channels, eps=1e-5)

  def forward(self, x):
    x = self.conv(x)
    x = self.gn(x)
    x = torch.cat([x, -x], 1)
    x = F.relu(x, inplace=True)
    return x

class RFE(nn.Module):
    """
            self.cls_subnet = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                RFE(256, 256),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
            self.box_subnet = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                RFE(256, 256),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
            )
            self.cls_subnet_pred = nn.Conv2d(256, num_anchors * cfg['num_classes'], kernel_size=1, stride=1, padding=0)
            self.box_subnet_pred = nn.Conv2d(256, num_anchors * 4, kernel_size=1, stride=1, padding=0)

    """ 
    def __init__(self, in_planes=256, out_planes=256):
        super(RFE, self).__init__()
        self.out_channels = out_planes
        self.inter_channels = int(in_planes / 4)

        self.branch0 = nn.Sequential(
                nn.Conv2d(in_planes, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(1, 5), stride=1, padding=(0, 2)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )
        self.branch1 = nn.Sequential(
                nn.Conv2d(in_planes, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(5, 1), stride=1, padding=(2, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )
        self.branch2 = nn.Sequential(
                nn.Conv2d(in_planes, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )
        self.branch3 = nn.Sequential(
                nn.Conv2d(in_planes, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.inter_channels, self.inter_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )
        self.cated_conv = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        out = self.cated_conv(out)
        out = out + x

        return out

class Backbone(nn.Module):
  """
  vgg backbone of ssd
  """
  def __init__(self):
    super(Backbone, self).__init__()
    layers = []
    # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
    #         512, 512, 512],
    layers += [nn.Conv2d(3, 64, kernel_size=3, padding=1)]     # conv1
    layers += [nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(64, 64, kernel_size=3, padding=1)]
    layers += [nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += [nn.Conv2d(64, 128, kernel_size=3, padding=1)]   # conv2
    layers += [nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(128, 128, kernel_size=3, padding=1)]
    layers += [nn.BatchNorm2d(128), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += [nn.Conv2d(128, 256, kernel_size=3, padding=1)]  # conv3
    layers += [nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
    layers += [nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1)]
    layers += [nn.BatchNorm2d(256), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

    layers += [nn.Conv2d(256, 512, kernel_size=3, padding=1)]  # conv4
    layers += [nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
    layers += [nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
    layers += [nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]  # conv5
    layers += [nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
    layers += [nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1)]
    layers += [nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]  # pool5

    layers += [nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)]  # conv6 (fc6)
    layers += [nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(1024, 1024, kernel_size=1)]  # conv7 (fc7)
    layers += [nn.BatchNorm2d(1024), nn.ReLU(inplace=True)]

  
  def add_extras(self, cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    # cfg =  [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    # 512:   (256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128),
    layers += [nn.Conv2d(1024, 256, kernel_size=1)]  # conv8
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    layers += [nn.Conv2d(512, 128, kernel_size=1)]  # conv9
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
    
    layers += [nn.Conv2d(256, 128, kernel_size=1)]  # conv10
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]
    
    layers += [nn.Conv2d(256, 128, kernel_size=1)]  # conv11
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    layers += [nn.Conv2d(256, 128, kernel_size=1)]  # conv12
    layers += [nn.Conv2d(128, 256, kernel_size=4, padding=1)]

    return layers