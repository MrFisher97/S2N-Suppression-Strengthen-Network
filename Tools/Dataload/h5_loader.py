import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch
import torch.utils.data as data_utl
from sklearn import preprocessing
import pandas as pd
# import h5py as h5
import numpy as np

SENSOR_SIZE = {
                "DVSGesture": (128, 128), 
                "DAVISGait":(260, 346), 
                "DAVISChar":(260, 346)
            }
TIME_SCALE = 1e6

class Base_Dataset(data_utl.Dataset):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self._collect(cfg)
        self.dataset = cfg.get('dataset', 'DVSGait')
        self.num_point = cfg.get('num_point', None)
        self.size = cfg.get('size', None)
        self.split_by = cfg.get('split_by', None)
 
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
            data = self.data[sample['light']][str(sample['label'])][sample['user']][sample['num']][:]
        elif self.dataset in ['DAVISGait', 'DAVISChar']:
            data = self.data[sample['light']][sample['obj']][sample['num']][:]
        
        data['t'] -= data['t'][0]
        data['t'] /= data['t'][-1]
        
        if self.cfg.get('reshape', False):
            center = (SENSOR_SIZE[self.dataset][0] // 2, SENSOR_SIZE[self.dataset][1] // 2)
            data = data[data['x'] > (center[0] - self.size[-2] // 2)]
            data = data[data['x'] < (center[0] + self.size[-2] // 2)]
            data = data[data['y'] > (center[1] - self.size[-1] // 2)]
            data = data[data['y'] < (center[1] + self.size[-1] // 2)]
            data['x'] -= (center[0] - self.size[-2] // 2)
            data['y'] -= (center[1] - self.size[-1] // 2)
        
        return data, sample['label']
    
    @staticmethod
    def collate_fn(batch):
        """
        Collects the different event representations and stores them together in a dictionary.
        """
        batch_dict = {}
        events = []
        labels = []
        for i, d in enumerate(batch):
            events.append(d['data'])
            labels.append(d['label'])
        batch_dict['data'] = torch.stack(events, 0)
        batch_dict['label'] = torch.tensor(labels)
        return batch_dict

    def __len__(self):
        return len(self.samples)

class Acc_Cnt_Clip(Base_Dataset):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        event, label = super().__getitem__(index)
        T, H, W = self.size
        
        t, x, y, p = event["t"], event["x"], event["y"], event["p"]
        x = np.array(x, dtype=np.long)

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

        return {'data':torch.tensor(clip, dtype=torch.float),
                'label':label}

class Acc_Cnt_Image(Base_Dataset):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        event, label = super().__getitem__(index)
 
        x, y, p = event["x"], event["y"], event["p"]
        H, W =  self.size
        
        x = x.astype(np.uint32)
        y = y.astype(np.uint32)
        p = p.astype(bool)

        img = np.zeros((2, H * W))
        np.add.at(img[0], x[p] + W * y[p], 1.)
        np.add.at(img[1], x[~p] + W * y[~p], 1.)

        img = img.reshape((2, H, W))

        # normalize along the space dimension
        img = np.divide(img, 
                        np.amax(img, axis=(-2, -1), keepdims=True),
                        out=np.zeros_like(img),
                        where=img!=0)

        return {'data':torch.as_tensor(img, dtype=torch.float),
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
        if self.num_point:
            idx = np.random.choice(data.shape[0], size = self.num_point, replace = False)
            data = data[idx]
        # data = resample(data, n_samples = self.num_point, random_state=2022)
        return {'data':torch.tensor(data),
                'label':label}
        
    @staticmethod
    def collate_fn(batch):
        """
        Collects the different event representations and stores them together in a dictionary.
        """
        batch_dict = {}
        events = []
        labels = []
        for i, d in enumerate(batch):
            ev = torch.concat([d['data'], i * torch.ones((len(d['data']),1), dtype=torch.float)], 1)
            events.append(ev)
            labels.append(d['label'])
        batch_dict['data'] = torch.concat(events, 0)
        batch_dict['label'] = torch.tensor(labels)
        return batch_dict


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
    dataset = Clip(cfg, transforms=None)

    # show the output
    # print(dataset[15]['vox'])
    import matplotlib.pyplot as plt
    canvas = np.zeros((260, 346, 3))
    canvas[..., :1] = np.array(dataset[29]['vox'][:, 2]).transpose(1, 2, 0)

    plt.figure()
    plt.imshow(canvas)
    plt.show()