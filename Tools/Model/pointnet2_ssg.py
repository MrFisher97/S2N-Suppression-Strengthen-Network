# import os, sys
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import PointNetSetAbstraction


class Model(nn.Module):
    def __init__(self, num_classes, seq_len = 1, normal_channel = False, return_fps=False, **kwargs):
        super(Model, self).__init__()
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=in_channel, mlp=[64, 64, 128], group_all=False, return_fps=return_fps)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False, return_fps=return_fps)
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        
        channels = np.array([1024, 512, 256]) * seq_len
        self.fc1 = nn.Linear(channels[0], channels[1])
        self.bn1 = nn.BatchNorm1d(channels[1])
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(channels[1], channels[2])
        self.bn2 = nn.BatchNorm1d(channels[2])
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(channels[2], num_classes)

    def forward(self, xyz):
        if len(xyz.shape) == 3:
            B, _, _ = xyz.shape
            S = 0
        elif len(xyz.shape) == 4:
            B, _, C, N = xyz.shape
            xyz = xyz.view(-1, C, N)
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points, l1_fps_idx = self.sa1(xyz, norm)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, -1)
        if self.training == False:
            points_set = [l1_xyz, l2_xyz, l3_xyz]
            features_set = [l1_points, l2_points, l3_points]
            fps_idx = [l1_fps_idx, l2_fps_idx]
            return x
        return x



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)
        return total_loss
