# # -- coding: utf-8 --**
import os
# import sys
# sys.path.append("..")

import numpy as np
import torch
import importlib
import argparse
import json
import Tools.utils as utils
import logging

class Test_Session(utils.Session):
    def build_log(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        self.logger = logger   
        self._build_model()

    def _build_model(self):
        super()._build_model()
        self.net = importlib.import_module(f"Tools.Model.{self.config['Model']['name']}").Model(**self.config['Model'])
        self.net = self.net.to(self.device)
        self.net.load_state_dict(torch.load(os.path.join(self.config['log_dir'], 'checkpoint.pkl')))

    @staticmethod
    def generate_clip(data, size=(2, 16, 128, 128), ord='txyp'):
        '''
        Generate Clip
        args:
            -data: event stream
            -size: size of generated clip (C, T, H, W)
            -ord: ording of event stream (e.g. 'txyp')
        output:
            clip
        '''

        t, x, y, p = np.split(data[:, (ord.find("t"), ord.find("x"), ord.find("y"), ord.find("p"))], 4, axis=1)
        C, T, H, W = size

        # normalization
        t -= np.min(t)
        t /= np.max(t)
        split_index = t * 0.99 * T
        
        x = x.astype(np.uint32)
        y = y.astype(np.uint32)
        p = p.astype(bool)
        split_index = split_index.astype(np.uint32)

        clip = np.zeros((C, T * H * W), dtype=np.float32)
        np.add.at(clip[0], x[p] + W * y[p] + H * W * split_index[p], 1)
        np.add.at(clip[1], x[~p] + W * y[~p] + H * W * split_index[~p], 1)
        
        clip = clip.reshape((C, T, H, W))
        clip = np.divide(clip, 
                        np.amax(clip, axis=(-2, -1), keepdims=True),
                        out=np.zeros_like(clip),
                        where=clip!=0)
        return clip

    def test_data(self, event):
        self.net.eval()
        with torch.no_grad():
            event = self.generate_clip(event, size=(2, 16, 128, 128), ord='txyp')
            event = torch.tensor(event[None, ...]).to(self.device)
            output = self.net(event)
            score = output['score'] if isinstance(output, dict) else output
        pred = score.argmax(1).item()
        return pred

    def test_dataset(self, data_loader):
        loss = utils.Param_Detector()
        acc = utils.Param_Detector()
        time = utils.Time_Detector()
        class_pred = utils.Category_Detector(self.config['Data']['num_classes'])
        
        for i, item in enumerate(data_loader):
            data = item['data'].to(self.device)
            output = self.net(data)
            score = output['score'] if isinstance(output, dict) else output
            pred = score.argmax(1).cpu()

            acc.update(pred.eq(item['label']).sum(), item['label'].size(0))
            time.update(item['label'].size(0))
            class_pred.update(pred, item['label'])
       
        return {'loss': loss.avg,
                'acc':acc.avg,
                'class_acc':class_pred.val,
                'time':time.avg}

    def test(self, scene='led'):
        self.net.eval()
        with torch.no_grad():
            self.config['Data']['scene'] = scene
            test_loader = self._load_data('Test')
            test_result = self.test_dataset(test_loader)
            self.logger.info(f"@ {scene}, loss:{test_result['loss']:.3f}, acc:{test_result['acc']:.1%}, {test_result['time']:.6f}  seconds/batch")


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default='DAVISGait_S2N')
    args.add_argument('--log_dir', type=str, default='Output/DAVISGait_S2N_2D_08292104')
    args = args.parse_args()
    config = json.load(open(f"Tools/Config/{args.config}.json", 'r'))
    config['log_dir'] = args.log_dir
    # exit(0)
    sess = Test_Session(config)
    sess.test('l64')
    # for scene in ['fluorescent', 'fluorescent_led', 'natural', 'led', 'lab']:
    #     sess.test(scene)