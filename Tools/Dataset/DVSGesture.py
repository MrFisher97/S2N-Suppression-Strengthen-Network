import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import glob
import torch
import torch.utils.data as data_utl
import numpy as np
np.random.seed(2021)
import Tools.Dataset.generator as generator
import stream_transforms
from sklearn.utils import resample
from sklearn import preprocessing
import pandas as pd
import h5py as h5

class Base_Dataset(data_utl.Dataset):
    def __init__(self, cfg, transforms=None, **kwargs):
        super(Base_Dataset, self).__init__()

        self._collect(cfg)
        self.transforms = transforms
 
    def _collect(self, cfg):
        root = cfg.get('root', 'Dataset/DVSGait/')
        data_file = cfg.get('data_file', 'C36W03.h5')
        label_map = cfg.get('map_file', 'map.csv')
        mode = cfg.get('mode', 'train')
        num_samples = cfg.get('num_samples', 1000)
        scene = cfg.get('scene', 'led')
        num_classes = cfg.get('num_classes', 10)

        samples = pd.read_csv(root + mode + '.csv', delimiter='\t')
        samples = samples[samples['light'] == scene] if scene != 'all' else samples
        samples = samples[:num_samples]
        samples = samples[samples['label'] < num_classes]

        assert len(samples) > 0, 'Error in Dataset!'
        self.samples = samples
        self.data = h5.File(root + data_file, 'r')
        self.label_map = pd.read_csv(root + label_map, delimiter='\t')

    def __len__(self):
        return len(self.samples)

class Frame_Dataset(data_utl.Dataset):
    def __init__(self,
                 root,
                 mode='train',
                 transforms=None,
                 nsample=None,
                 scene=None,
                 size=(16, 2, 128, 128),
                 num_classes=10,
                 ordering='txyp',
                 split_by='time',
                 **kwargs):
        super(Frame_Dataset, self).__init__()


        assert len(size) == 4, 'Wrong size'

        self.files = self._collect(root, mode, scene, num_classes, nsample)
        self.clip_size = size
        self.transforms = transforms
        self.ordering = ordering
        self.split_by = split_by
        self.convertor = generator.generate_clip
        # if in_channels == 2:
        #     self.convertor = event_to_frame.generate_two_channels_clip_by_time
        # elif in_channels == 4:
        #     self.convertor = event_to_frame.generate_four_channels_clip_by_time

    def _collect(self, root, mode, scene, num_classes, nsample):
        if scene == 'all' or scene == None:
            fileList = glob.glob(os.path.join(root, mode, '*'))
        else:
            fileList = glob.glob(os.path.join(root, mode, f'*user[0-9][0-9]_{scene}_N*'))
        fileList.sort()

        files = []
        for f in fileList:
            l = int(f.split('/')[-1].split('_')[0]) - 1
            if l < num_classes:
                files.append(f)
        files = files[:nsample]
        assert len(files) > 0, 'Wrong dataset root!'
        return files

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        stream = np.load(self.files[index]).astype(np.float64) # stream size:[n, 4] ; 4:t, x, y, p
        # stream[:, 0] = (stream[:, 0] - np.min(stream[:, 0])) / (np.max(stream[:, 0]) - np.min(stream[:, 0]))

        if self.transforms is not None:
            stream = self.transforms(stream)
        
        stream[:, 0] = preprocessing.minmax_scale(stream[:, 0])
        clip = self.convertor(stream,
                              ordering=self.ordering,
                              clip_size=self.clip_size,
                              split_by=self.split_by) # clip shape : t * x * y * c
        label = int(self.files[index].split('/')[-1].split('_')[0]) - 1

        return {'clip':torch.tensor(clip, dtype=torch.float).permute(3, 0, 1, 2),
                'label':label}

    def __len__(self):
        return len(self.files)

class PCloud_Dataset(data_utl.Dataset):
    def __init__(self,
                 root,
                 mode='train',
                 transforms=None,
                 nsample=None,
                 scene=None,
                 npoint=1024,
                 size=(15, 128, 128),
                 num_classes=10,
                 **kwargs):
        super(PCloud_Dataset, self).__init__()
        if scene == 'all' or scene == None:
            files = glob.glob(os.path.join(root, mode, '*'))
        else:
            files = glob.glob(os.path.join(root, mode, f'*user[0-9][0-9]_{scene}_N*'))

        self.files = []
        for f in files:
            l = int(f.split('/')[-1].split('_')[0]) - 1
            if l < num_classes:
                self.files.append(f)

        self.npoint = npoint
        self.transforms = transforms
        self.size = size
        self.files = self.files[:nsample]

    def downsample(self, stream):
        if self.npoint > stream.shape[0]:
            stream = resample(stream, replace=True, n_samples=self.npoint)
        idx = np.arange(stream.shape[0])
        np.random.shuffle(idx)
        idx = idx[0:self.npoint]
        return stream[idx,...]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        stream = np.load(self.files[index]).astype(np.float64) # stream size:[n, 4] ; 4:t, x, y, p
        stream[:, 0] = preprocessing.minmax_scale(stream[:, 0])
        
        if self.transforms is not None:
            stream = self.transforms(stream)

        pcloud = self.downsample(stream)
        pcloud[:, 1] = pcloud[:, 1] / self.size[-2]
        pcloud[:, 2] = pcloud[:, 2] / self.size[-1]
        label = int(self.files[index].split('/')[-1].split('_')[0]) - 1

        return {'clip':torch.tensor(pcloud[:, :3], dtype=torch.float).permute(1, 0),
                'label':label}

    def __len__(self):
        return len(self.files)

if __name__  == '__main__':
    pass