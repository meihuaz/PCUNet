import os
import sys
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def read_pcd(filename):
    pcd = o3d.io.read_point_cloud(filename)
    return np.array(pcd.points)


def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size=n - pcd.shape[0])])
    return pcd[idx[:n]]


class PCNDataset(Dataset):
    def __init__(self, root, input_size, gt_size, split='train'):
        self.root = root
        self.input_size = input_size
        self.gt_size = gt_size
        self.datapath = []
        self.filepath = os.path.join(self.root, split, 'partial')
        for root, dirs, files in os.walk(self.filepath):
            break
        for dir in dirs:
            filedir = os.path.join(self.filepath, dir)
            for root, dirs1, files in os.walk(filedir):
                subdirs = dirs1
                break
            for subdir in subdirs:
                subdir = os.path.join(filedir, subdir)
                datapath = [X_f for X_f in sorted(os.listdir(subdir))]
                for X_f in datapath:
                    self.datapath.append(os.path.join(subdir, X_f))

    def __getitem__(self, index):
        return self._get_item(index)

    def _get_item(self, index):
        pc_path = self.datapath[index]
        partial = read_pcd(pc_path)
        partial = resample_pcd(partial, self.input_size)

        pc_complete_path = pc_path.replace('partial', 'complete')
        pc_complete_path = pc_complete_path[:-7] + ".pcd"
        complete = read_pcd(pc_complete_path)
        complete = resample_pcd(complete, self.gt_size)
        return partial, complete

    def __len__(self):
        return len(self.datapath)


class PCNCarDataset(Dataset):
    def __init__(self, root, input_size, gt_size, split='train'):
        self.root = root
        self.input_size = input_size
        self.gt_size = gt_size
        self.datapath = []
        self.filepath = os.path.join(self.root, split, 'partial', '02958343')
        for root, dirs, files in os.walk(self.filepath):
            break
        for dir in dirs:
            filedir = os.path.join(self.filepath, dir)
            datapath = [X_f for X_f in sorted(os.listdir(filedir))]
            for X_f in datapath:
                self.datapath.append(os.path.join(filedir, X_f))

    def __getitem__(self, index):
        return self._get_item(index)

    def _get_item(self, index):
        pc_path = self.datapath[index]
        partial = read_pcd(pc_path)
        partial = resample_pcd(partial, self.input_size)

        pc_complete_path = pc_path.replace('partial', 'complete')
        pc_complete_path = pc_complete_path[:-7] + ".pcd"
        complete = read_pcd(pc_complete_path)
        complete = resample_pcd(complete, self.gt_size)
        return partial, complete

    def __len__(self):
        return len(self.datapath)


class PCNDatasetTest(Dataset):
    def __init__(self, root, input_size, gt_size, classchoice='Plane'):
        self.root = root
        self.input_size = input_size
        self.gt_size = gt_size

        if classchoice == "Plane":
            classchoicecode = '02691156'
        if classchoice == "Cabinet":
            classchoicecode = '02933112'
        if classchoice == "Car":
            classchoicecode = '02958343'
        if classchoice == "Chair":
            classchoicecode = '03001627'
        if classchoice == "Lamp":
            classchoicecode = '03636649'
        if classchoice == "Couch":
            classchoicecode = '04256520'
        if classchoice == "Table":
            classchoicecode = '04379243'
        if classchoice == "Watercraft":
            classchoicecode = '04530566'

        self.datapath = []
        datapath = self.root + 'test' + '/complete/' + classchoicecode
        data_files = [X_f for X_f in sorted(os.listdir(datapath))]
        for path in data_files:
            self.datapath.append(datapath + '/' + path)

    def __getitem__(self, index):
        return self._get_item(index)

    def _get_item(self, index):
        pc_path = self.datapath[index]
        complete = read_pcd(pc_path)
        complete = resample_pcd(complete, self.gt_size)
        pc_path_partial = pc_path.replace('complete', 'partial')
        partial = read_pcd(pc_path_partial)
        partial = resample_pcd(partial, self.input_size)
        return partial, complete, pc_path

    def __len__(self):
        return len(self.datapath)


class PCNDatasetNovelTest(Dataset):
    def __init__(self, root, input_size, gt_size, classchoice='bench'):
        self.root = root
        self.input_size = input_size
        self.gt_size = gt_size

        if classchoice == "bench":
            classchoicecode = '02828884'
        if classchoice == "Skateboard":
            classchoicecode = '04225987'
        if classchoice == "Pistol":
            classchoicecode = '03948459'
        if classchoice == "Motorbike":
            classchoicecode = '03790512'
        if classchoice == "Guitar":
            classchoicecode = '03467517'
        if classchoice == "bed":
            classchoicecode = '02818832'
        if classchoice == "bookshelf":
            classchoicecode = '02871439'
        if classchoice == "bus":
            classchoicecode = '02924116'

        self.datapath = []
        datapath = self.root + 'test_novel' + '/complete/' + classchoicecode
        data_files = [X_f for X_f in sorted(os.listdir(datapath))]
        for path in data_files:
            self.datapath.append(datapath + '/' + path)

    def __getitem__(self, index):
        return self._get_item(index)

    def _get_item(self, index):
        pc_path = self.datapath[index]
        complete = read_pcd(pc_path)
        complete = resample_pcd(complete, self.gt_size)
        pc_path_partial = pc_path.replace('complete', 'partial')
        partial = read_pcd(pc_path_partial)
        partial = resample_pcd(partial, self.input_size)
        return partial, complete, pc_path

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    dset = PCNDataset(root="/root/shenzhen/zmh/point_cloud/dataset/pcn_dataset/",
                      input_size=3000, gt_size=16384, split='train')
    # dset = PCNDatasetTest(root="/root/shenzhen/zmh/point_cloud/dataset/pcn_dataset/",
    #                       input_size=3000, gt_size=16384, classchoice='Plane')
    print(len(dset))
    loader = DataLoader(dset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    for i, batch in enumerate(loader):
        print(i)
