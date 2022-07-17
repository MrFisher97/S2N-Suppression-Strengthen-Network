import torch
import torch.nn as nn
import torch.nn.functional as F

MIN_VALUE = 1e-6

class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, Depth_conv=nn.Conv2d):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = Depth_conv(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(InceptionModule, self).__init__()

        self.b0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels>>2,
            kernel_size=1,
        )

        self.b1 = CSDN_Tem(in_ch=in_channels,  out_ch=out_channels>>1, kernel_size=3)
        self.b2 = CSDN_Tem(in_ch=in_channels,  out_ch=out_channels>>2, kernel_size=5)

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(x)
        b2 = self.b2(x)
        x = torch.cat([b0,b1,b2], dim=1)
        return x


class Model(nn.Module):
    def __init__(self, scale_factor, in_channels):
        super(Model, self).__init__()

        self.activation = nn.Tanh()
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 16
        self.enhance_iter = 2
        self.mean_val = 0
        kernel_size = 3
        self.lconv1 = CSDN_Tem(in_channels, number_f, kernel_size=kernel_size)
        self.lconv2 = InceptionModule(number_f, number_f<<1, kernel_size=kernel_size) 
        self.mconv =  nn.Sequential(
                        CSDN_Tem(number_f<<1, number_f<<2, kernel_size=kernel_size),
                        self.activation,
                        CSDN_Tem(number_f<<2, number_f<<1, kernel_size=kernel_size),
                        self.activation,
                        )
        self.rconv1 = InceptionModule(number_f<<2, number_f, kernel_size=kernel_size) 
        self.rconv2 = CSDN_Tem(number_f<<1, in_channels, kernel_size=kernel_size) 


    def enhance(self, x, x_r):
        enhance_res = [x]
        # momentun = 0.5
        # x_mean = torch.mean(x[x > 0])
        # if self.training:
        #     self.mean_val = momentun * x_mean + (1 - momentun) * self.mean_val
        # self.enhance_iter = torch.clamp(self.mean_val / x_mean, min=1, max=4)
        # for i in range(int(self.enhance_iter.item())):
        for i in range(self.enhance_iter):
            x = enhance_res[-1]
            enhance_res.append(x + x_r * (x - 1) *  x)
            # enhance_res.append(x + x_r * (torch.sin(x)))
        return enhance_res[-1]
		
    def forward(self, x):
        if self.scale_factor==1:
            x_down = x
        else:
            x_down = F.interpolate(x,scale_factor=1/self.scale_factor, mode='bilinear')

        x1 = self.activation(self.lconv1(x_down))
        x2 = self.activation(self.lconv2(x1))

        x3 = self.mconv(x2)

        x2 = self.activation(self.rconv1(torch.cat([x2,x3],1)))
        x1 = torch.tanh(self.rconv2(torch.cat([x1, x2],1)))
        if self.scale_factor==1:
            E = x1
        else:
            E = self.upsample(x1)
        x = self.enhance(x, E)  
        return x, E


class L_channel(nn.Module):
    '''
        Channel consistency loss
        In event frame, there exists two channel, positive and negative.
    '''
    def __init__(self):
        super(L_channel, self).__init__()

    def forward(self, mask):
        mask = torch.mean(mask, dim=(2, 3))
        diff = torch.pow(mask[:, 0] - mask[:, 1], 2)
        return torch.mean(diff)

class L_noise(nn.Module):
    '''
        Noise loss
        We coarsely identify the noise if it lacks adjacent event in k*k neighbor
    '''
    def __init__(self, k):
        super(L_noise, self).__init__()
        self.k = k
        kernel = torch.ones((1, 2, k, k)).cuda()
        kernel[..., k // 2, k // 2] = 0
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, enhance):
        bool_enhance = (enhance > MIN_VALUE).to(torch.float)
        around = F.conv2d(bool_enhance, self.weight, padding=self.k // 2, groups=1).repeat(1, 2, 1, 1)
        mean = self.avgpool(bool_enhance)
        mask = (around < 100 * mean) & (enhance > 0)

        area = torch.sum(mask, dim=(2, 3))
        noise = torch.sum(enhance * mask, dim=(2, 3))
        return torch.mean(noise / area), mask

class L_exp(nn.Module):
    '''
        Exposure loss
        We hope the intensity of event frame to be strong enough while keeping the local contrast as much as possible
    '''
    def __init__(self,patch_size):
        super(L_exp, self).__init__()
        self.avgpool = nn.AvgPool2d(patch_size)

    def forward(self, raw, enhance, noise_mask):
        valid_mask = ~noise_mask & (enhance > 0)
        
        enhance_mean = self.avgpool(enhance * valid_mask)
        enhance_mean = enhance_mean[enhance_mean > MIN_VALUE]
        
        expected_mean = torch.clamp(torch.pow(enhance_mean, 0.5), min=.6)
        exp_loss = torch.mean(torch.pow(enhance_mean - expected_mean, 2))
        return exp_loss


class NSN_Loss(nn.Module):
    def __init__(self, patch_size=5, E=0.7, n=1):
        super(NSN_Loss, self).__init__()
        self.noise = L_noise(5)
        self.channel = L_channel()
        self.exp = L_exp(patch_size)
        self.n = n
        self.E = E

    def forward(self, enhanced_image, A, raw_iamges):
        loss_noise, noise_mask = self.noise(enhanced_image)
        loss_noise *= self.n
        loss_channel = self.channel(A)
        loss_exp = 5 * self.exp(raw_iamges, enhanced_image, noise_mask)
    
        loss = 0
        loss += loss_noise
        loss += loss_channel
        loss += loss_exp          

        return {'loss':{
                    'total':loss,
                    'noise':loss_noise,
                    'channel':loss_channel,
                    'exposure':loss_exp,
                    },
                'mask': ~noise_mask & (enhanced_image > 0),
                }