import torch
import torch.nn as nn
from utils import *


class SARB(nn.Module):
    def __init__(self, c_in, c_out, relu_type='relu', norm_type='in', hg_depth=2, att_name='spar'):
        super(SARB, self).__init__()
        self.c_in      = c_in
        self.c_out     = c_out
        self.norm_type = norm_type
        self.relu_type = relu_type
        self.hg_depth  = hg_depth

        kwargs = {'norm_type': norm_type, 'relu_type': relu_type}
        
        self.conv1 = ConvLayer(c_in, c_out, 3, **kwargs) 
        self.conv2 = ConvLayer(c_out, c_out, 3, norm_type=norm_type, relu_type='none')

        self.att_func = SEDN(self.hg_depth, c_out, 1) 
        
    def forward(self, x):
        identity = x 
        out = self.conv1(x)
        out = self.conv2(out)
        out = identity + self.att_func(out)
        return out
        

class SEDN(nn.Module):
    def __init__(self, depth, c_in, c_out, c_mid=64,norm_type='in', relu_type='relu'):
        super(SEDN, self).__init__()
        self.depth     = depth
        self.c_in      = c_in
        self.c_mid     = c_mid
        self.c_out     = c_out
        self.kwargs = {'norm_type': norm_type, 'relu_type': relu_type}

        if self.depth:
            self._generate_network(self.depth)
            self.out_block = nn.Sequential(
                    ConvLayer(self.c_mid, self.c_out, norm_type='none', relu_type='none'),
                    nn.Sigmoid()
                    )
            
    def _generate_network(self, level):
        if level == self.depth:
            c1, c2 = self.c_in, self.c_mid
        else:
            c1, c2 = self.c_mid, self.c_mid

        self.add_module('b1_' + str(level), ConvLayer(c1, c2, **self.kwargs)) 
        self.add_module('b2_' + str(level), ConvLayer(c1, c2, scale='down', **self.kwargs)) 
        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvLayer(self.c_mid, self.c_mid, **self.kwargs)) 

        self.add_module('b3_' + str(level), ConvLayer(self.c_mid, self.c_mid, scale='up', **self.kwargs))

    def _forward(self, level, in_x):
        up1 = self._modules['b1_' + str(level)](in_x)
        low1 = self._modules['b2_' + str(level)](in_x)
        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = self._modules['b2_plus_' + str(level)](low1)

        up2 = self._modules['b3_' + str(level)](low2)
        if up1.shape[2:] != up2.shape[2:]:
            up2 = nn.functional.interpolate(up2, up1.shape[2:])
        
        return up1 + up2

    def forward(self, x, pmask=None):
        if self.depth == 0: return x
        input_x = x
        x = self._forward(self.depth, x)
        self.att_map = self.out_block(x)
        x = input_x * self.att_map 
        return x 


class Generator(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, c_dim=5, repeat_num=1):
        super(Generator, self).__init__()

        layers = []
        layers.append(nn.Conv2d(3+c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-sampling layers.
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        layers.append(SARB(c_in=curr_dim, c_out=curr_dim))

        # Up-sampling layers.
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True, track_running_stats=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        layers.append(nn.Conv2d(curr_dim, 3, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())     
        self.att_func = SEDN(2, 3, 1) 
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        xo = x
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)        
        y = self.main(x)
        x = y + xo
        return x

