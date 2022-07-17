import torch
import torch.nn as nn
# from Backbone import gaitimg
from Enhance import Filter, NSN
import ed2d as backbone

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        num_classes = config['Data'].get('num_classes', 10)
        in_channels = config['Model'].get('in_channels', 2)
        nIter = config['Model'].get('nIter', 3)

        self.nsn = NSN.Model(scale_factor=1, in_channels=in_channels) if config['Model']['enhance'] else None
        
        self.fsn = nn.ModuleDict(
            {
                'EAM': Filter.Anistropic_Diffusion(in_channels=in_channels, kernel_size=1, sigma=1, nIter=nIter),
                'DCDC':  backbone.Model(num_classes=num_classes, in_channels=in_channels),
            
            }
        )


    def forward(self, x):
        enhance = x
        if self.enhance:
            with torch.no_grad():
                enhance, _ = self.nsn(x)
        x = self.fsn['EAM'](enhance)
        score = self.fsn['DCDC'](x)
        return {'score':score, 'enhance':enhance}