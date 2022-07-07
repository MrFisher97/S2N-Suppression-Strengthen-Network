import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

def normalize(x, dim):
    u = torch.mean(x, dim=dim, keepdim=True)
    std = torch.std(x, dim=dim, keepdims=True)
    x = (x - u) / std
    return x


def standard(x, dim):
    x_min = torch.amin(x, dim=dim, keepdim=True)
    x_max = torch.amax(x, dim=dim, keepdims=True)
    x = (x - x_min) / (x_max - x_min)
    return x

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.kernel_size = kernel_size
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim
 
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        # self.register_buffer('weight', kernel)
        self.weight = nn.Parameter(kernel, requires_grad=False)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
            self.dim = (2)
        elif dim == 2:
            self.conv = F.conv2d
            self.dim = (2, 3)
        elif dim == 3:
            self.conv = F.conv3d
            self.dim = (2, 3, 4)
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        x = self.conv(input, weight=self.weight, groups=self.groups, padding=self.kernel_size//2)
        return normalize(x, self.dim)

class Anistropic_Diffusion(nn.Module):
    def __init__(self, in_channels, sigma=0.1, nIter=3, **kwargs):
        super(Anistropic_Diffusion, self).__init__()
        direction_weights = []

        self.max_sigma = sigma
        self.num_direction = 8
        self.in_channels = in_channels
        self.nIter = nIter

        if self.num_direction == 4:
            direction = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        elif self.num_direction == 8:
            direction = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for i in range(self.num_direction):
            kernel = torch.zeros((3, 3)) 
            kernel[1, 1] = -1
            kernel[1 + direction[i][0], 1 + direction[i][1]] = 1
            direction_weights.append(kernel)

        direction_weights = torch.stack(direction_weights, dim=0)
        direction_weights = direction_weights.unsqueeze(1).tile(in_channels, 1, 1, 1)

        self.weight = nn.Parameter(direction_weights, requires_grad=False)
        # self.weight = direction_weights
        conv_weight = torch.ones((2, self.num_direction, 1, 1)) / 4
        self.conv_weight = nn.Parameter(conv_weight, requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):

        # get the gradient
        xGrad = F.conv2d(x, weight=self.weight, padding=1, groups=self.in_channels)

        # Get the sigma
        sigma = self.pool(torch.abs(xGrad))
        B, C = sigma.size()[:2]
        sigma = sigma.view(B, C//self.in_channels, self.in_channels, 1, 1).softmax(dim=1)

        # clamp the sigma
        sigma = torch.clamp(sigma.view(B, C, 1, 1) * 8, min=0.01, max=self.max_sigma)

        for i in range(self.nIter):
            gWeight = torch.exp(- torch.pow(xGrad, 2) / sigma)
            diffusion = xGrad * gWeight
            diffusion = F.conv2d(diffusion, weight=self.conv_weight, groups=self.in_channels)
            diffusion[diffusion < 0] = 0
            x = x + diffusion
            x = standard(x, dim=(2, 3))
            if self.nIter > 1:
                xGrad = F.conv2d(x, weight=self.weight, padding=1, groups=self.in_channels)
        return normalize(x, dim=(2, 3))
        # return x