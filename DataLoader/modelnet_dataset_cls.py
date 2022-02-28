'''
    ModelNet dataset. Support ModelNet40, ModelNet10, XYZ and normal channels. Up to 10000 points.
'''

import os
import os.path
import json
import numpy as np
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class ModelNetDataset(Dataset):
    def __init__(self, root, npoints=1024, split='train', normalize=True, normal_channel=False, modelnet10=False):
        self.root = root
        self.npoints = npoints
        self.normalize = normalize
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]

    def _get_item(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        # cls = np.array([cls]).astype(np.int32)
        point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
        # Take the first npoints
        point_set = point_set[0:self.npoints, :]
        if self.normalize:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.normal_channel:
            point_set = point_set[:, 0:3]

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)

    def __len__(self):
        return len(self.datapath)

    def num_channel(self):
        if self.normal_channel:
            return 6
        else:
            return 3


if __name__ == '__main__':
    # import time
    # time_start = time.time()

    modelnet_dataset = ModelNetDataset(root=os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled'), split='test')
    loader = DataLoader(modelnet_dataset, batch_size=10, shuffle=True, num_workers=1)
    for _, batch in enumerate(loader):
        ps_batch = batch[0]
        cls_batch = batch[1]
        print(ps_batch)
        print(cls_batch)

    #
    #
    #
    #
    # print(d.has_next_batch())
    # ps_batch, cls_batch = d.next_batch(True)
    # print(ps_batch.shape)
    # print(cls_batch.shape)
