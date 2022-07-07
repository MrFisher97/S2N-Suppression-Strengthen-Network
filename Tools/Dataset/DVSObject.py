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
from sklearn.utils import resample
from sklearn import preprocessing

class Base_Dataset(data_utl.Dataset):
    def __init__(self,
                 root,
                 mode='train',
                 transforms=None,
                 nsample=None,
                 scene=None,
                 num_classes=35,
                 label_file='char_mapping.csv',
                 **kwargs):
        super(Base_Dataset, self).__init__()

        with open(label_file, 'r') as f:
            self.labelDict = {s.split(',')[0].strip():int(s.split(',')[1].strip())
                                for s in f.readlines()[1:]}
        
        self.files = self._collect(root, mode, scene, num_classes, nsample)
        self.transforms = transforms
        self.root = root
 
    def _collect(self, root, mode, scene, num_classes, nsample):
        with open(os.path.join(root, mode+'.csv'), 'r') as f:
            files = f.readlines()

        fileList = []
        for f in files:
            identity = f.split('/')[0]
            light = f.split('/')[1].split('_')[0]
            label = int(self.labelDict[identity])
            if label < num_classes:
                if (light == scene) or (scene == 'all'):
                    fileList.append(f.strip())
        fileList = fileList[:nsample]
        assert len(fileList) > 0, 'Wrong dataset root!'
        return fileList

    def __len__(self):
        return len(self.files)

class Pic_Dataset(Base_Dataset):
    def __init__(self,
                 root,
                 mode='train',
                 transforms=None,
                 nsample=None,
                 scene=None,
                 size=(1, 2, 260, 346),
                 num_classes=35,
                 ordering='txyp',
                 split_by='time',
                 label_file='identity_mapping.csv',
                 **kwargs):

        super(Pic_Dataset, self).__init__(root, mode, transforms, 
                                            nsample, scene, num_classes,
                                            label_file,)
        assert len(size) == 4, 'Wrong size'
        self.pic_size = size
        self.ordering = ordering
        self.split_by = split_by
        self.convertor = generator.generate_clip
        # if in_channels == 2:
        #     self.convertor = event_to_frame.generate_two_channels_clip_by_time
        # elif in_channels == 4:
        #     self.convertor = event_to_frame.generate_four_channels_clip_by_time

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = os.path.join(self.root, self.files[index])
        stream = np.load(path).astype(np.float64) # stream size:[n, 4] ; 4:t, x, y, p
        # stream[:, 0] = (stream[:, 0] - np.min(stream[:, 0])) / (np.max(stream[:, 0]) - np.min(stream[:, 0]))

        if self.transforms is not None:
            stream = self.transforms(stream)
        stream[:, 0] = preprocessing.minmax_scale(stream[:, 0])
        clip = self.convertor(stream,
                              ordering=self.ordering,
                              clip_size=self.pic_size,
                              split_by=self.split_by) # clip shape : t * x * y * c
        label = self.labelDict[path.split('/')[-2]]

        return {'clip':torch.tensor(clip, dtype=torch.float)[0].permute(2, 0, 1),
                'label':label}

class Frame_Dataset(Base_Dataset):
    def __init__(self,
                 root,
                 mode='train',
                 transforms=None,
                 nsample=None,
                 scene=None,
                 size=(16, 2, 260, 346),
                 num_classes=35,
                 ordering='txyp',
                 split_by='time',
                 label_file='identity_mapping.csv',
                 **kwargs):

        super(Frame_Dataset, self).__init__(root, mode, transforms, 
                                            nsample, scene, num_classes,
                                            label_file,)
        assert len(size) == 4, 'Wrong size'
        self.clip_size = size
        self.ordering = ordering
        self.split_by = split_by
        self.convertor = generator.generate_clip
        # if in_channels == 2:
        #     self.convertor = event_to_frame.generate_two_channels_clip_by_time
        # elif in_channels == 4:
        #     self.convertor = event_to_frame.generate_four_channels_clip_by_time

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = os.path.join(self.root, self.files[index])
        stream = np.load(path).astype(np.float64) # stream size:[n, 4] ; 4:t, x, y, p
        # stream[:, 0] = (stream[:, 0] - np.min(stream[:, 0])) / (np.max(stream[:, 0]) - np.min(stream[:, 0]))

        if self.transforms is not None:
            stream = self.transforms(stream)
        stream[:, 0] = preprocessing.minmax_scale(stream[:, 0])
        clip = self.convertor(stream,
                              ordering=self.ordering,
                              clip_size=self.clip_size,
                              split_by=self.split_by) # clip shape : t * x * y * c
        label = self.labelDict[path.split('/')[-2]]

        return {'clip':torch.tensor(clip, dtype=torch.float).permute(3, 0, 1, 2),
                'label':label}

class PCloud_Dataset(Base_Dataset):
    def __init__(self,
                 root,
                 mode='train',
                 transforms=None,
                 nsample=None,
                 scene=None,
                 npoint=1024,
                 size=(15, 128, 128),
                 num_classes=35,
                 label_file='identity_mapping.csv',
                 **kwargs):
        
        super(PCloud_Dataset, self).__init__(root, mode, transforms, 
                                            nsample, scene, num_classes,
                                            label_file,)
        self.npoint = npoint
        self.size = size

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
        path = os.path.join(self.root, self.files[index])
        stream = np.load(path).astype(np.float64) # stream size:[n, 4] ; 4:t, x, y, p
        stream[:, 0] = preprocessing.minmax_scale(stream[:, 0])
        
        if self.transforms is not None:
            stream = self.transforms(stream)

        pcloud = self.downsample(stream)
        pcloud[:, 1] = pcloud[:, 1] / self.size[-2]
        pcloud[:, 2] = pcloud[:, 2] / self.size[-1]
        label = self.labelDict[path.split('/')[-2]]

        return {'clip':torch.tensor(pcloud[:, :3], dtype=torch.float).permute(1, 0),
                'label':label}

if __name__  == '__main__':
    pass