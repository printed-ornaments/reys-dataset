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
        
        # Background extraction
        # (B, 3, H, W), (B, 3, H, W), (B,)
        bg_likelihood, bg, kl_bg, log_bg = self.bg_module(x, global_step)
        
        # Foreground extraction
         fg_likelihood, fg, alpha_map, kl_fg, log_fg, original_image, virtual_loss, z_where, obj_prob, z_cls = self.gmair_fg(x, global_step)
       
        # Fix alpha trick
        if global_step and global_step < arch.fix_alpha_steps:
            alpha_map = torch.full_like(alpha_map, arch.fix_alpha_value)
            #alpha_map_1 = torch.full_like(alpha_map_1, arch.fix_alpha_value)
        
        

        fg_likelihood = (fg_likelihood + (alpha_map + 1e-5).log())
        bg_likelihood = (bg_likelihood + (1 - alpha_map + 1e-5).log())
        # (B, 2, 3, H, W)
        log_like = torch.clamp(torch.stack((fg_likelihood, bg_likelihood), dim=1),-10)
        # (B, 3, H, W)
        log_like = torch.logsumexp(log_like, dim=1)
        # (B,)
        log_like = log_like.flatten(start_dim=1).sum(1)

        # Take mean as reconstruction
        fg_bg = torch.full_like(bg, 255)
        y = (alpha_map)*fg +  ((1.0 - alpha_map)) * bg
        fg_seg = (alpha_map)*fg +  ((1.0 - alpha_map)) * fg_bg
        
        '''y=y*255
        original_image*=25'''
        print(alpha_map)

        if global_step % 1 == 0:
           fin_img = transform(y[0])
           fin_img_fg = transform(((fg_seg))[0])
           fin_img_bg = transform(bg[0])
           fin_orig = transform(x[0])
           st_full = '../results/examples_SPAGMACE/gmair/'+str(global_step)+'_new_full'+'.png'
           st_fg ='../results/examples_SPAGMACE/gmair/'+str(global_step)+'_new_fg'+'.png'
           st_bg ='../results/examples_SPAGMACE/gmair/'+str(global_step)+'_new_bg'+'.png'
           st_bg ='../results/examples_SPAGMACE/gmair/'+str(global_step)+'_new_bg'+'.png'            
           st_ori ='../results/examples_SPAGMACE/gmair/'+str(global_step)+'_ori'+'.png'            
           fin_img.save(st_full)
           fin_img_fg.save(st_fg)
           fin_img_bg.save(st_bg)
           fin_orig.save(st_ori)
        # Elbo
        elbo = log_like - kl_bg - kl_fg
        
        # Mean over batch
        loss = (-elbo+virtual_loss).mean()        
        x1 = [(log_fg['z_pres'] >0.87).nonzero(as_tuple=True)]
        log = {
            'imgs': x,
            'y': y,
            # (B,)
            'mse': ((y-x)**2).flatten(start_dim=1).sum(dim=1),
            'log_like': log_like
        }
        log.update(log_fg)
        log.update(log_bg)
        
        return loss, log
