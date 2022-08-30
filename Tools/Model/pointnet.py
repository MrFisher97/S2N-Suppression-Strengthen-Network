import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from pointnet_utils import PointNetEncoder, feature_transform_reguliarzer

class Pointnet(nn.Module):
    def __init__(self, num_classes=10, normal_channel=False):
        super(Pointnet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x.float())
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # x = F.log_softmax(x, dim=1)
        return x

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        # mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        # total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return loss

class Model(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.model = Pointnet(num_classes=num_classes)

    def forward(self, x):
        x = x.transpose(2, 1)[:, :3]
        x[:, (1, 2)] /= 127
        return self.model(x)
