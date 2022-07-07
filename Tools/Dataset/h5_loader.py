import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch
import torch.utils.data as data_utl
from generator import Generator
from sklearn.utils import resample
from sklearn import preprocessing
import pandas as pd
# import h5py as h5
import numpy as np

class Base_Dataset(data_utl.Dataset):
    def __init__(self, cfg, transforms=None, **kwargs):
        super().__init__()
        self._collect(cfg)
        self.dataset = cfg.get('dataset', 'DVSGait')
        self.num_point = cfg.get('num_point', None)
        self.clip_size = cfg.get('clip_size', None)
        self.vox_size = cfg.get('vox_size', None)
        self.split_by = cfg.get('split_by', None)
        self.transforms = transforms
 
    def _collect(self, cfg):
        root = cfg.get('root', 'Dataset/DVSGait/')
        data_file = cfg.get('data_file', 'C36W03.h5')
        file_list = cfg.get('file_list', 'train.csv')
        label_map = cfg.get('map_file', 'map.csv')
        num_samples = cfg.get('num_samples', 1000)
        scene = cfg.get('scene', 'led')
        num_classes = cfg.get('num_classes', 10)

        samples = pd.read_csv(root + file_list, delimiter='\t')
        samples = samples if scene in ['all', None] else samples[samples['light'] == scene]
        samples = samples[:num_samples]
        samples = samples[samples['label'] < num_classes]

        assert len(samples) > 0, 'Error in Dataset!'
        self.samples = samples.reset_index(drop=True)
        import h5py
        self.data = h5py.File(root + data_file, 'r')
        self.label_map = pd.read_csv(root + label_map, delimiter='\t')
        self.ord = cfg.get('ord', 'txyp')

    def __getitem__(self, index):
        sample = self.samples.loc[index]
        if self.dataset == 'DVSGesture':
            data = self.data[sample['light']][str(sample['label'])][sample['user']][sample['num']]
        elif self.dataset in ['DVSGait', 'DVSChar']:
            data = self.data[sample['light']][sample['obj']][sample['num']]
        
        data = np.stack([data[p] for p in self.ord], axis=-1).astype(float)
        if self.transforms is not None:
            data = self.transforms(data)
        data[:, 0] = preprocessing.minmax_scale(data[:, 0])
        return data, sample['label']

    def __len__(self):
        return len(self.samples)

class Clip(Base_Dataset):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        event, label = super().__getitem__(index)
 
        t, x, y, p = np.split(event[:, (self.ord.find("t"), self.ord.find("x"), self.ord.find("y"), self.ord.find("p"))], 4, axis=1)
        T, H, W =  self.size

        if self.split_by == 'time':
            t = t * 0.99 * T
        elif self.split_by == 'cnt':
            t = np.arange(0, 1, 1/event.shape[0]) * T
        
        x = x.astype(np.uint32)
        y = y.astype(np.uint32)
        p = p.astype(bool)
        split_index = t.astype(np.uint32)

        clip = np.zeros((2, T * H * W))
        np.add.at(clip[0], x[p] + W * y[p] + H * W * split_index[p], 1.)
        np.add.at(clip[1], x[~p] + W * y[~p] + H * W * split_index[~p], 1.)

        clip = clip.reshape((2, T, H, W))

        # normalize along the space dimension
        clip = np.divide(clip, 
                        np.amax(clip, axis=(2, 3), keepdims=True),
                        out=np.zeros_like(clip),
                        where=clip!=0)

        return {'data':torch.tensor(clip),
                'label':label}

class Voxel(Base_Dataset):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        data, label = super().__getitem__(index)
        vox = Generator.generate_voxel(data, ord=self.ord, vox_size=self.vox_size)
        return {'data':torch.tensor(vox),
                'label':label}

class Point(Base_Dataset):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        data, label = super().__getitem__(index)
        data = resample(data, n_samples = self.num_point, random_state=2022)
        data[:, 0] = preprocessing.minmax_scale(data[:, 0])
        return {'data':torch.tensor(data).to(self.device),
                'label':label.to(self.device)}

# Test class
if __name__ == '__main__':
    cfg = {
        'dataset':'DVSGait',
        'root':'Dataset/DVSGait/',
        'data_file':'C36W03.h5',
        'map_file':'map.csv',
        'mode':'train',
        'num_samples': 1000,
        'scene': 'l64',
        'num_classes':10,
        'num_point':2048,
        'clip_size':(4, 5, 260, 346),
        'vox_size':(1, 5, 260, 346),
        'split_by':'time',
    }
    dataset = Voxel(cfg, transforms=None)

    # show the output
    # print(dataset[15]['vox'])
    import matplotlib.pyplot as plt
    canvas = np.zeros((260, 346, 3))
    canvas[..., :1] = np.array(dataset[29]['vox'][:, 2]).transpose(1, 2, 0)

    plt.figure()
    plt.imshow(canvas)
    plt.show()