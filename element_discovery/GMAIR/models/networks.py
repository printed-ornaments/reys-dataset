from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, SELU, Linear, Module, ModuleDict, ModuleList
from torch.nn import functional as F

from gmair.config import config as cfg
from gmair.utils import debug_tools

class BasicNetwork(Module):
    def __init__(self, n_in_channels, n_out_channels, topology, internal_activation=SELU):
        '''
        Builds CNN
        :param n_in_channels:
        :param n_out_channels:
        :param topology:
        :param internal_activation:
        :return:
        '''
        super().__init__()
        self.topology = topology
        
        self.net = self._build_backbone(n_in_channels, n_out_channels)

    def _build_backbone(self, n_in_channels, n_out_channels):
        '''Builds the convnet of the backbone'''

        n_prev = n_in_channels
        net = OrderedDict()

        # Builds internal layers except for the last layer
        for i, layer in enumerate(self.topology):
            layer['in_channels'] = n_prev

            if 'filters' in layer.keys():
                f = layer.pop('filters')
                layer['out_channels'] = f #rename
            else:
                f = layer['out_channels']

            net['conv_%d' % i] = Conv2d(**layer)
            net['act_%d' % i] = SELU()
            n_prev = f

        # Builds the final layer
        net['conv_out'] = Conv2d(in_channels=f, out_channels=n_out_channels, kernel_size=1, stride=1)
                
        return Sequential(net)

    def forward(self, x):
        x = self.net(x)
        return x
        
def get_heat_map_head():
    n_features = cfg.n_backbone_features
    
    n_class = 1
    
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        ]
     
    return BasicNetwork(n_features, n_class, topology)
    
def get_where_head():
    n_features = cfg.n_backbone_features
    
    n_localization_latent = 8  # mean and var for (y, x, h, w)
    
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        ]
     
    return BasicNetwork(n_features, n_localization_latent, topology)
        
def get_depth_head():
    n_features = cfg.n_backbone_features

    n_depth_latent = 2
    
    topology = [
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        dict(filters=128, kernel_size=3, stride=1, padding=1),
        ]
     
    return BasicNetwork(n_features, n_depth_latent, topology)


# [B, chan * obj_size * obj_size + num_classes + 1] -> [B, 2 * N_ATTR] 
class ObjectEncoder(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(ObjectEncoder, self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SELU(),
            nn.Linear(512, 512),
            nn.SELU(),
            nn.Linear(512, 512),
            nn.SELU(),
            nn.Linear(512, output_dim),
        ) 
        
    
    def forward(self, x):  
        x = self.linear(x)
        return x


# [B, N_ATTR] -> [B, (chan + 1) * obj_size * obj_size]
class ObjectDecoder(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(ObjectDecoder, self).__init__()
    
        self.linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SELU(),
            nn.Linear(128, 256),
            nn.SELU(),
            nn.Linear(256, 512),
            nn.SELU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        x = self.linear(x)
        return x
'''
# [B, C, 32, 32] -> [B, 2 * N_ATTR] 
class ObjectConvEncoder(nn.Module):
    
    def __init__(self, input_channels):
        super(ObjectConvEncoder, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.GroupNorm(4, 16),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 32),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 64),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 128, 4, 2, 1),
            nn.ReLU(),
            nn.GroupNorm(8, 128),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        
        self.linear = nn.Linear(256, 2 * cfg.N_ATTRIBUTES)
    
    def forward(self, x):
        
        x = self.cnn(x)
        flat_x = x.flatten(start_dim = 1)
        out = self.linear(flat_x)
        
        return out

'''

# [B, N_ATTR] -> [B, C, 32, 32]
class ObjectConvDecoder(nn.Module):
    
    def __init__(self, out_channels):
        super(ObjectConvDecoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(cfg.n_what, 256),
            nn.SELU(),
        )
        
        self.cnn = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.Conv2d(256, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.SELU(),
            nn.GroupNorm(8, 128),
            # nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.Conv2d(128, 128 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.SELU(),
            nn.GroupNorm(8, 128),
            # nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.Conv2d(128, 64 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.SELU(),
            nn.GroupNorm(8, 64),
            # nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.Conv2d(64, 32 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.SELU(),
            nn.GroupNorm(8, 32),
            # nn.ConvTranspose2d(32, 32, 3, 1, 1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.SELU(),
            nn.GroupNorm(8, 32),
            # nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.Conv2d(32, 16 * 2 * 2, 1),
            nn.PixelShuffle(2),
            nn.SELU(),
            nn.GroupNorm(4, 16),
            # nn.ConvTranspose2d(16, out_channels, 3, 1, 1),
            nn.Conv2d(16, out_channels, 3, 1, 1),
            
            # nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 256, 1, 1)
        out = self.cnn(x)
    
        return out
