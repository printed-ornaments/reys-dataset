''' Modified based on https://github.com/yonkshi/SPAIR_pytorch '''

from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, Linear, Module, ModuleDict, ModuleList
from torch.nn import functional as F

from gmair.config import config as cfg
from gmair.utils import debug_tools


def latent_to_mean_std(latent_var, dim = -1):
    '''
    Converts a VAE latent vector to mean and std. log_std is converted to std.
    :param latent_var: VAE latent vector
    :return:
    '''
    mean, log_std = torch.chunk(latent_var, 2, dim = dim)
    # std = log_std.mul(0.5).exp_()
    # std = torch.sigmoid(log_std.clamp(-10, 10)) * 2
    std = F.softplus(log_std)
    return mean, std

def clamped_sigmoid(logit, use_analytical=False):
    '''
    Sigmoid function,
    :param logit:
    :param use_analytical: use analytical sigmoid function to prevent backprop issues in pytorch
    :return:
    '''
    logit = torch.clamp(logit, -10, 10)
    if use_analytical:
        return 1 / ((-logit).exp() + 1)

    return torch.sigmoid(torch.clamp(logit, -10, 10))
    
def linear_decay(global_step:float, device, start, end, decay_step:float, staircase=False):
    '''
    A decay helper function for computing decay of
    :param global_step:
    :param start:
    :param end:
    :param decay_rate:
    :param decay_step:
    :param staircase:
    :return:
    '''
    global_step = torch.tensor(global_step, dtype=torch.float32).to(device)
    if staircase:
        t = global_step // decay_step
    else:
        t = global_step / decay_step
    
    value = end if t > 1 else (start - end) * (1 - t) + end

    return value
    
def exponential_decay(global_step:float,device, start, end, decay_rate, decay_step:float, staircase=False, log_space=False, ):
    '''
    A decay helper function for computing decay of
    :param global_step:
    :param start:
    :param end:
    :param decay_rate:
    :param decay_step:
    :param staircase:
    :param log_space:
    :return:
    '''
    global_step = torch.tensor(global_step, dtype=torch.float32).to(device)
    if staircase:
        t = global_step // decay_step
    else:
        t = global_step / decay_step
    value = (start - end) * (decay_rate ** t) + end

    if log_space:
        value = (value + 1e-6).log()

    return value


def stn(image, z_where, output_dims, device, inverse=False):
    """
    Slightly modified based on https://github.com/kamenbliznashki/generative_models/blob/master/air.py

    spatial transformer network used to scale and shift input according to z_where in:
            1/ x -> x_att   -- shapes (H, W) -> (attn_window, attn_window) -- thus inverse = False
            2/ y_att -> y   -- (attn_window, attn_window) -> (H, W) -- thus inverse = True
    inverting the affine transform as follows: A_inv ( A * image ) = image
    A = [R | T] where R is rotation component of angle alpha, T is [tx, ty] translation component
    A_inv rotates by -alpha and translates by [-tx, -ty]
    if x' = R * x + T  -->  x = R_inv * (x' - T) = R_inv * x - R_inv * T
    here, z_where is 3-dim [scale, tx, ty] so inverse transform is [1/scale, -tx/scale, -ty/scale]
    R = [[s, 0],  ->  R_inv = [[1/s, 0],
         [0, s]]               [0, 1/s]]
    """

    xt, yt, xs, ys = torch.chunk(z_where, 4, dim=-1)
    yt = yt.squeeze()
    xt = xt.squeeze()
    ys = ys.squeeze()
    xs = xs.squeeze()

    batch_size = image.size(0)
    color_chans = cfg.input_image_shape[0]
    out_dims = [batch_size, color_chans] + output_dims # [Batch, RGB, obj_h, obj_w]

    # Important: in order for scaling to work, we need to convert from top left corner of bbox to center of bbox
    yt = (yt ) * 2 - 1
    xt = (xt ) * 2 - 1

    theta = torch.zeros(2, 3).repeat(batch_size, 1, 1).to(device)

    # set scaling
    theta[:, 0, 0] = xs
    theta[:, 1, 1] = ys
    # set translation
    theta[:, 0, -1] = xt
    theta[:, 1, -1] = yt

    # inverse == upsampling
    if inverse:
        # convert theta to a square matrix to find inverse
        t = torch.tensor([0., 0., 1.]).repeat(batch_size, 1, 1).to(device)
        t = torch.cat([theta, t], dim=-2)
        t = t.inverse()
        theta = t[:, :2, :]
        out_dims = [batch_size, color_chans + 1] + output_dims  # [Batch, RGBA, obj_h, obj_w]

    # 2. construct sampling grid
    grid = F.affine_grid(theta, out_dims, align_corners=False)

    # 3. sample image from grid
    padding_mode = 'border' if not inverse else 'zeros'
    input_glimpses = F.grid_sample(image, grid, padding_mode=padding_mode, align_corners=False)
    # debug_tools.plot_stn_input_and_out(input_glimpses)


    return input_glimpses

