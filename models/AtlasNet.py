from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math
from torch.autograd import Variable
from utils.model_utils import gen_grid_up, calc_emd, calc_cd
import numpy as np
import time


class STN3d(nn.Module):
    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, num_points=8192, global_feat=True):
        super(PointNetfeat, self).__init__()
        self.stn = STN3d(num_points=num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(self.bottleneck_size // 2, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class Model(nn.Module):
    def __init__(self, args, bottleneck_size=1024):
        super(Model, self).__init__()
        self.train_loss = args.loss
        self.num_points = args.num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = args.n_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(self.num_points, global_feat=True),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList(
            [PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])

    def forward(self, x, gt, is_training=True, mean_feature=None, alpha=None):

        if mean_feature:
            raise NotImplementedError
        partial = x
        x = self.encoder(x)
        outs = []
        for i in range(0, self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.n_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))

        outs = torch.cat(outs, 2).contiguous()
        out1 = outs.transpose(1, 2).contiguous()
        out2 = out1

        if is_training:
            if self.train_loss == 'emd':
                loss1 = 0
                loss2 = calc_emd(out2, gt)
            elif self.train_loss == 'cd':
                loss1 = 0
                loss2, _ = calc_cd(out2, gt)
            else:
                raise NotImplementedError('Train loss is either CD or EMD!')

            total_train_loss = loss2.mean()
            return out2, loss2, total_train_loss
        else:
            emd = calc_emd(out2, gt, eps=0.004, iterations=3000)
            cd_p, cd_t, f1 = calc_cd(out2, gt, calc_f1=True)
            return {'out1': out1, 'out2': out2, 'emd': emd, 'cd_p': cd_p, 'cd_t': cd_t, 'f1': f1}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv2d') == -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and classname.find('BatchNorm2d') == -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
