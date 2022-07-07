import time
import os
import numpy as np
import logging
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import importlib

class Session():
    def __init__(self, config):
        self.config = config
        self.build_log()
        
    def build_log(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        timestamp = time.strftime('%m%d%H%M', time.localtime(time.time()))
        recorder = self.config['Recorder']
        self.config['timestamp'] = timestamp

        if recorder['save_log']:
            self.log_dir = os.path.join(recorder['log_dir'], timestamp)
            os.makedirs(self.log_dir, exist_ok=True)
            os.system(f"cp -n Tools/Config/{self.config['Data']['dataset']}.yaml {self.log_dir}")
            log_file = os.path.join(self.log_dir, 'logger.txt')
            fh = logging.FileHandler(filename=log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s : %(message)s", "%b%d-%H:%M"))
        else:
            log_file = 'temp_log.txt'
            fh = logging.FileHandler(filename=log_file)
            fh.setFormatter(logging.Formatter("%(asctime)s : %(message)s", "%b%d-%H:%M"))
        
        if not logger.handlers:
            logger.addHandler(ch)
            logger.addHandler(fh)

        self.logger = logger
        self.writer = SummaryWriter('Runs') if recorder['show_tensorboard'] else None
        
        self._build_model()

    def _build_model(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enable = True
        
        torch.manual_seed(2022)
        torch.cuda.manual_seed(2022)
        torch.cuda.manual_seed_all(2022)
        
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

    def _load_data(self, mode, num_samples=None):

        batch_size = self.config[mode].get('batch_size', 16)
        shuffle = self.config[mode].get('shuffle', False)
        num_workers = self.config[mode].get('num_workers', 0)
        drop_last = self.config[mode].get('drop_last', False)
        form = self.config['Data'].get('form', 'Voxel')
        self.config['Data']['file_list'] = self.config[mode]['file_list']
        dataset = getattr(importlib.import_module("Tools.Dataset.h5_loader"), form)(self.config['Data'])
        dataset.samples = dataset.samples[:num_samples]
        loader = DataLoader(dataset, 
                            batch_size=batch_size,
                            shuffle=shuffle, 
                            num_workers=num_workers, 
                            drop_last=drop_last,)

        data_file = self.config['Data'].get('data_file')
        scene = self.config['Data'].get('scene', 'all')
        self.logger.info(f'{mode} Dataset size {len(dataset)} @{data_file}_{scene} has been loaded')
        return loader

    def close(self):
        if self.writer:
            self.writer.close()

class Param_Detector():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class Time_Detector():
    """Computes the average occurrence of some event per second"""
    def __init__(self, init=0):
        self.reset(init)

    def reset(self, init=0):
        self.init = init
        self.start = time.time()
        self.n = 0

    def update(self, n=1):
        self.n += n

    @property
    def avg(self):
        return self.elapsed_time / self.n

    @property
    def elapsed_time(self):
        return self.init + (time.time() - self.start)

class Param_Dict():
    def __init__(self):
        self.param_dict = {}
    
    def reset(self):
        for k, v in self.param_dict.items():
            self.param_dict = v.reset()
    
    def update(self, val_dict, n=1):
        for k, v in val_dict.items():
            if k not in self.param_dict.keys():
                self.param_dict[k] = Param_Detector()
            v = v.detach().cpu()
            self.param_dict[k].update(v)

    def __getitem__(self, k):
        if k not in self.param_dict.keys():
            self.param_dict[k] = Param_Detector()
        return self.param_dict[k]

    def items(self):
        return self.param_dict.items()

class Category_Detector():
    """Stores the prediction category"""
    def __init__(self, num_class):
        self.num_class = num_class
        self.reset()

    def reset(self):
        self.val = np.zeros((self.num_class, self.num_class))

    def update(self, pred, label):
        for i, p in enumerate(pred):
            self.val[label[i], pred[i]] += 1