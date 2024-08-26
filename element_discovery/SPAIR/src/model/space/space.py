import torch
import torch.nn as nn
import torch.nn.functional as F
from .arch import arch
from .fg import SpaceFg
from .bg import SpaceBg
from .gmair.models.model import gmair
from .tensorboardX import SummaryWriter
from .debug_tools import *
device = 'cpu'
import torchvision.transforms as T
transform = T.ToPILImage()

class Space(nn.Module):
    
    def __init__(self, image_shape, writer:SummaryWriter, device):
        nn.Module.__init__(self)
        
        self.fg_module = SpaceFg()
        self.bg_module = SpaceBg()
        self.gmair_fg =  gmair(image_shape, writer, device)
        
    def forward(self, x, global_step):
        """
        Inference.
        
        :param x: (B, 3, H, W)
        :param global_step: global training step
        :return:
            loss: a scalor. Note it will be better to return (B,)
            log: a dictionary for visualization
            
        """
        
        # Foreground extraction
        fg_likelihood, fg, alpha_map, kl_fg, loss_boundary, log_fg = self.fg_module(x, global_step)        

        # Fix alpha trick
        if global_step and global_step < arch.fix_alpha_steps:
            alpha_map = torch.full_like(alpha_map, arch.fix_alpha_value)        
        

        fg_likelihood = (fg_likelihood + (alpha_map + 1e-5).log())
        log_like = torch.logsumexp(fg_likelihood, dim=1)
        log_like = log_like.flatten(start_dim=1).sum(1)

        y = fg
        # Elbo
        elbo = log_like - kl_fg
        
        # Mean over batch
        loss = (-elbo+loss_boundary).mean()        
        x1 = [(log_fg['z_pres'] >0.87).nonzero(as_tuple=True)]
        log = {
            'imgs': x,
            'y': y,
            # (B,)
            'mse': ((y-x)**2).flatten(start_dim=1).sum(dim=1),
            'log_like': log_like
        }
        log.update(log_fg)
        
        return loss, log
