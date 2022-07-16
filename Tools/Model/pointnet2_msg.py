import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pointnet_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class Pointnet2_msg(nn.Module):
    def __init__(self, num_classes=10, seq_len = 1, normal_channel = False, return_fps=False, **kwargs):
        super(Pointnet2_msg, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        
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
            xyz = xyz[:, :3, :]
            norm = None

        l1_xyz, l1_points, l1_fps_idx = self.sa1(xyz, norm)
        l2_xyz, l2_points, l2_fps_idx = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, -1)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        if self.training == False:
            points_set = [l1_xyz, l2_xyz, l3_xyz]
            features_set = [l1_points, l2_points, l3_points]
            fps_idx = [l1_fps_idx, l2_fps_idx]
            return x
        return x


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Pointnet2_msg(num_classes=config["Data"]["num_classes"])

    def forward(self, x):
        x = x.transpose(2, 1)
        return self.model(x)