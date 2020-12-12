import torch
from itertools import product as product
import numpy as np
from math import ceil
from math import sqrt


class PriorBox_sar(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox_sar, self).__init__()
        self.aspect_ratios = cfg['aspect_ratios']  # [[0.5, 2], [0.5, 2], [0.5, 2], [0.5, 2]]
        self.min_sizes = cfg['min_sizes']  # [[8, 16], [32, 64, 128], [256], [512]]
        self.steps = cfg['steps']  # [16, 32, 64, 128]
        self.clip = cfg['clip']  # False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                    cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                    for ar in self.aspect_ratios[k]:
                        anchors += [cx, cy, s_kx*sqrt(ar), s_ky/sqrt(ar)]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class PriorBox_sar_old(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox_sar_old, self).__init__()
        self.aspect_ratios = cfg['aspect_ratios']  # [[0.5, 2], [0.5, 2], [0.5, 2], [0.5, 2]]
        self.min_sizes = cfg['min_sizes']  # [[8, 16], [32, 64, 128], [256], [512]]
        self.steps = cfg['steps']  # [16, 32, 64, 128]
        self.clip = cfg['clip']  # False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 8:  # 8 times
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.125, j+0.25, j+0.375, j+0.5, j+0.625, j+0.75, j+0.875]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.125, i+0.25, i+0.375, i+0.5, i+0.625, i+0.75, i+0.875]]
                        for cy, cx in product(dense_cy, dense_cx):
                            for ar in self.aspect_ratios[k]:
                                anchors += [cx, cy, s_kx*sqrt(ar), s_ky/sqrt(ar)]
                    elif min_size == 16 or min_size == 32:  # 4 times
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            for ar in self.aspect_ratios[k]:
                                anchors += [cx, cy, s_kx*sqrt(ar), s_ky/sqrt(ar)]
                    elif min_size == 64:  # 2 times
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            for ar in self.aspect_ratios[k]:
                                anchors += [cx, cy, s_kx*sqrt(ar), s_ky/sqrt(ar)]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        for ar in self.aspect_ratios[k]:
                            anchors += [cx, cy, s_kx*sqrt(ar), s_ky/sqrt(ar)]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output

class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        #self.aspect_ratios = cfg['aspect_ratios']
        self.min_sizes = cfg['min_sizes']  # [[32, 64, 128], [256], [512]]
        self.steps = cfg['steps']  # [32, 64, 128]
        self.clip = cfg['clip']  # False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    if min_size == 32:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.25, j+0.5, j+0.75]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.25, i+0.5, i+0.75]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    elif min_size == 64:
                        dense_cx = [x*self.steps[k]/self.image_size[1] for x in [j+0, j+0.5]]
                        dense_cy = [y*self.steps[k]/self.image_size[0] for y in [i+0, i+0.5]]
                        for cy, cx in product(dense_cy, dense_cx):
                            anchors += [cx, cy, s_kx, s_ky]
                    else:
                        cx = (j + 0.5) * self.steps[k] / self.image_size[1]
                        cy = (i + 0.5) * self.steps[k] / self.image_size[0]
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class PriorBox_ssd(object):
    # Compute priorbox coordinates in center-offset form for each source
    # feature map.
    def __init__(self, cfg):
        super(PriorBox_ssd, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output