import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['L2Norm', 'RFE']


class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

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