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
            data = self.data[sample['light']][str(sample['label'])][sample['user']][sample['num']]
        elif self.dataset in ['DAVISGait', 'DAVISChar']:
            data = self.data[sample['light']][sample['obj']][sample['num']]
        
        data = np.stack([data[p] for p in self.ord], axis=-1).astype(float)
        
        if self.cfg.get('reshape', False):
            event = self.reshape_event(event, self.cfg, self.ord, senseor_size=SENSOR_SIZE[self.dataset])
        
        data[:, 0] = preprocessing.minmax_scale(data[:, 0])
        return data, sample['label']

    @staticmethod
    def reshape_event(event, cfg, ord='txyp', senseor_size=(128, 128)):
        method = cfg.get('reshape_method', 'no')
        new_size = cfg.get('size', (224, 224))[-2:]
        if method == 'sample':
            sampling_ratio = np.pord(new_size) / np.pord(senseor_size)
            new_len = int(sampling_ratio * len(event))
            idx_arr = np.arange(len(event))
            sampled_arr = np.random.choice(idx_arr, size=new_len, replace=False)
            event = event[np.sort(sampled_arr)]

        event[:, ord.find('x')] *= (new_size[0] / senseor_size[0])
        event[:, ord.find('y')] *= (new_size[1] / senseor_size[1])

        if method == 'unique':
            coords = event[:, (ord.find('x'), ord.find('y'))].astype(np.int64)
            timestamp = (event[:,  ord.find('t')] * TIME_SCALE).astype(np.int64)
            min_time = timestamp[0]
            timestamp -= min_time

            key = coords[:, 0] + coords[:, 1] * new_size[1] + timestamp * np.prod(new_size)
            _, unique_idx = np.unique(key, return_index=True)
            event = event[unique_idx]

        event[:, (ord.find('x'), ord.find('y'))] = event[:, (ord.find('x'), ord.find('y'))].astype(np.int64)

        return event

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

        return {'data':torch.tensor(clip, dtype=torch.float),
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
            events.append(d['data'])
            labels.append(d['label'])
        batch_dict['data'] = torch.stack(events, 0)
        batch_dict['label'] = torch.tensor(labels)
        return batch_dict

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