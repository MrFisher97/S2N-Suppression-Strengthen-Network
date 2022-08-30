import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DCD_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.1):

        super(DCD_Conv2d, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta
        self.kernel_shape = (kernel_size, kernel_size) if type(kernel_size) == int else kernel_size

    def forward(self, x):
        out_normal = self.conv(x)

        kernel_diff = self.conv.weight.sum((2, 3))
        kernel_diff = kernel_diff[:, :, None, None]

        pool_pad = list(map(lambda x: x // 2, self.kernel_shape))
        mask = F.avg_pool2d((x > 0).to(torch.float16), kernel_size=self.kernel_shape,
                        padding=pool_pad, stride=1)                       

        out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        return out_normal - self.theta * out_diff

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            DCD_Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            DCD_Conv2d(outchannel, outchannel, kernel_size=3, padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()

    def forward(self, x):
        # print("x:" + str(x.shape))
        out = self.left(x)
        # print("out: " +str(out.shape))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Model(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()

        self.features = nn.Sequential(
            DCD_Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            DCD_Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            DCD_Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            DCD_Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.AdaptiveAvgPool2d((14)),
        )

        self.classifier = nn.Sequential(
            # nn.Dropout(),
            # nn.Linear(num_channel*16*5*5,128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(128,10),
            nn.Linear(14*14*512, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )


    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x, 1)
        # x = x[0].view(-1, self.fc1.weight.size(1))
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    image = torch.randn([1,4,128,128])
    net = Model(10, 4)
    net(image)
    # print(net.named_modules)