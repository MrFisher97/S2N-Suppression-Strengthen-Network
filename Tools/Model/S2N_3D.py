import torch
import torch.nn as nn
import ed3d as backbone
from Enhance import Filter, NSN

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        num_classes = config['Data'].get('num_classes', 10)
        in_channels = config['Model'].get('in_channels', 2)
        nIter = config['Model'].get('nIter', 1)

        self.nsn = NSN.Model(scale_factor=1, in_channels=in_channels) if config['Model']['enhance'] else None
        self.fsn = nn.ModuleDict(
            {
                'EAM': Filter.Anistropic_Diffusion(in_channels=in_channels, kernel_size=1, sigma=1, nIter=nIter),
                'DCDC':  backbone.Model(num_classes=num_classes, in_channels=in_channels),
            
            }
        )

    def forward(self, x):
        B, C, T, H, W = x.size()
        enhance = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        if self.nsn:
            with torch.no_grad():
                enhance, _ = self.nsn(enhance)
        x = self.fsn['EAM'](enhance)
        x = x.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        score = self.fsn['DCDC'](x)
        return {'score':score, 'enhance':enhance}
