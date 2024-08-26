import itertools
import argparse
import math
import numpy as np
from ipdb import set_trace
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, ReLU, Linear
from torch.nn import functional as F
from torch.distributions import Normal, Uniform, Poisson
from torch.distributions.kl import kl_divergence
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import math
from gmair.config import config as cfg
from .modules import *
from .networks import *
from .resnet_dcn import get_resnet_dcn
from .gumble_softmax import GumbelSoftmax
from numpy import unique
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from numpy import where

from sklearn.mixture import GaussianMixture

class gmair(nn.Module):
    def __init__(self, image_shape, writer:SummaryWriter, device):
        super().__init__()
        self.image_shape = image_shape
        
        self.writer = writer
        self.device = device

        self._build_networks()
        self._build_priors()

        self.pixels_per_cell = (image_shape[1] / self.feature_space_dim[1], image_shape[2] / self.feature_space_dim[2])
        #self.pixels_per_cell = 16.0
        print(self.pixels_per_cell)
        print(self.pixels_per_cell)
        
        print('model initialized')
        
    def _build_networks(self):
        
        #print("Entered Build Network Block")

        # backbone network
        # [C, 128, 128] -> [C', 32, 32]
        self.backbone = get_resnet_dcn(18)
        self.feature_space_dim = self.backbone.output_shape()
        
        # pres & bounding box & depth
        self.heat_map_head = get_heat_map_head()
        self.where_head = get_where_head()
        self.depth_head = get_depth_head()
        
        # gumbel softmax
        self.gumbel_softmax = GumbelSoftmax()
        
        # BN-VAE
        _, H, W = self.feature_space_dim
        #print(H)
        self.box_mu_bn = nn.BatchNorm1d(H * W * 4)
        self.box_mu_bn.weight.requires_grad = False
        self.box_mu_bn.weight.fill_(cfg.box_mu_bn_gamma)
        nn.init.constant_(self.box_mu_bn.bias, 0.0)
        
        # what prior : p(z^{what} | z^{cls})
        self.what_mu_prior = nn.Linear(cfg.num_classes, cfg.n_what) # , bias = False)
        self.what_var_prior = nn.Linear(cfg.num_classes, cfg.n_what) # , bias = False)
        
        # object encoder
        c = cfg.input_image_shape[0]
        h, w = cfg.object_shape
        in_dim = c * h * w # + cfg.num_classes
        self.object_encoder_what = ObjectEncoder(in_dim + cfg.num_classes, 2 * cfg.n_what)
        self.object_encoder_cat = ObjectEncoder(in_dim, cfg.num_classes)
          
        # object decoder
        # [GrayScale + Alpha] or [RGB+A]
        if cfg.decoder_type == 'mlp':
            out_dim = (c + 1) * h * w
            self.object_decoder = ObjectDecoder(cfg.n_what, out_dim)
        elif cfg.decoder_type == 'cnn':
            self.object_decoder = ObjectConvDecoder(c + 1)
            
        
    def _build_priors(self):
        '''
        builds independ priors for the VAE for KL computation (VAE KL Regularizer)
        '''
        
        #print("Build Priors Block")
        self.kl_priors = {}
        for z_name, (mean, std) in cfg.priors.items():
            dist = Normal(mean, std)
            self.kl_priors[z_name] = dist
            
    def _update_hyperparams(self, step):
        self.global_step = step
        self.tau = 1.0
        self.prior_obj = np.maximum(np.exp(-cfg.obj_cls_prior_k * step), cfg.obj_cls_prior_bottom)
        self.log_prior_obj = np.maximum(-cfg.obj_cls_prior_k * step, np.log(cfg.obj_cls_prior_bottom))
        
        # --- record hyperparameters ---
        self.writer.add_scalar('hyperparams/tau', self.tau, self.global_step)
        self.writer.add_scalar('hyperparams/prior_obj', self.prior_obj, self.global_step)
        
    def forward(self, x, global_step = -1, mode = 'train'):
        
        assert mode == 'train' or mode == 'infer'
        
        # debug_tools.benchmark_init()
        self.mode = mode
        if mode == 'train':
            self._update_hyperparams(global_step)
            
        # keeps track of all distributions
        self.dist = {}

        # feat : [batch, C, H, W]
        feat = self.backbone(x)
        
        # cls_logits : [batch, H, W, num_classes + 1]
        pres_logits = self.heat_map_head(feat).permute(0,2,3,1).contiguous()

        # where_latent : [batch, H, W, 8]
        where_latent = self.where_head(feat).permute(0,2,3,1).contiguous()
        
        # depth_latent : [batch, H, W, 2]
        depth_latent = self.depth_head(feat).permute(0,2,3,1).contiguous()
        
        # z_where : [batch, H, W, 4]
        z_where, mean, std = self._build_where(where_latent)
        
        # z_depth : [batch, H, W, 1]
        z_depth = self._build_depth(depth_latent)
        
        # z_what : [batch, H, W, N_WHAT]
        _, H, W = self.feature_space_dim
        #print(H)
        input_glimpses = stn(x.repeat_interleave(H*W, dim = 0), z_where.view(-1, 4), cfg.object_shape, self.device)
        print(cfg.object_shape)
        print(cfg.num_classes)
        input_glimpses = input_glimpses.flatten(start_dim = 1)
        cls_logits = self.object_encoder_cat(input_glimpses).view(-1,H,W,cfg.num_classes)
        
        # z_cls : [batch, H, W, num_classes] (gumbel softmax)
        # prob : [batch, H, W, num_classes] (softmax)
        # obj_prob : [batch, H, W]
        z_cls, prob, obj_prob = self._build_cls(pres_logits, cls_logits)
        
        concat = torch.cat([input_glimpses, z_cls.view(-1,cfg.num_classes)], dim = -1)
        what_latent = self.object_encoder_what(concat).view(-1,H,W,2*cfg.n_what)
        
        z_what = self._build_what(what_latent)
        
        #print("The SIZE OF Z WHQT IS")
        #print(z_where)

        if self.mode == 'infer':
            # renderer p(x|z)
            recon_x, z_where = self._render(z_what, z_where, z_depth, obj_prob, x)   
            return recon_x, z_cls, z_what, z_where, obj_prob
                
        kl_loss = self._compute_KL(cls_logits, z_cls, prob, obj_prob)
        recon_x, virtual_loss, objects_mask, objects = self._render(z_what, z_where, z_depth, obj_prob, x)
        loss = self._build_loss(x, recon_x, kl_loss, virtual_loss, objects_mask, objects, obj_prob)
        

        
        return loss, recon_x

    def _build_cls(self, pres_logits, cls_logits):
        ''' Builds the network to detect object presence'''
        
        # [batch, H, W, num_classes]
        # prob = torch.softmax(cls_logits, dim = -1)
        # z_cls = prob
        
        if self.mode == 'train':
            prob, z_cls = self.gumbel_softmax(cls_logits, temperature=self.tau, hard=False)
        else:
            prob = torch.softmax(cls_logits, dim = -1)
            z_cls = prob
            
        obj_prob = torch.sigmoid(pres_logits).flatten(start_dim = 2)
 
        return z_cls, prob, obj_prob

    # where_latent_var : [batch, H, W, 8]
    def _build_where(self, where_latent):
        ''' Builds z_where '''

        mean, std = latent_to_mean_std(where_latent)

        
        #print('mean is: '+str(mean.size()))
        #print('std is: '+str(std.size()))

        # BN-VAE
        _, H, W = self.feature_space_dim
        mean = mean.contiguous().view(-1, H * W * 4)
        #print(mean.size())
        mean = self.box_mu_bn(mean)
        mean = mean.view(-1, H, W, 4)
        

        # [batch, H, W, 1]
        offset_y_mean, offset_x_mean, scale_y_mean, scale_x_mean = torch.chunk(mean, 4, dim=-1)
        offset_y_std, offset_x_std, scale_y_std, scale_x_std = torch.chunk(std, 4, dim=-1)
        
        offset_y_logits = self._sample_z(offset_y_mean, offset_y_std, 'offset_y')
        offset_x_logits = self._sample_z(offset_x_mean, offset_x_std, 'offset_x')
        scale_y_logits = self._sample_z(scale_y_mean, scale_y_std, 'scale_y')
        scale_x_logits = self._sample_z(scale_x_mean, scale_x_std, 'scale_x')


        # --- cell y/x transform ---
        # yx [-0.5, 0.5]
        cell_y = clamped_sigmoid(offset_y_logits) - 0.5
        cell_x = clamped_sigmoid(offset_x_logits) - 0.5

        # --- height/width transform ---
        # hw ~ [0.0 , 1.0]
        height = clamped_sigmoid(scale_y_logits)
        width = clamped_sigmoid(scale_x_logits)
        
   
        box_max_shape = cfg.box_max_shape
        _, image_height, image_width = cfg.input_image_shape
        
        # bounding box height & width relative to the whole image
        ys = height * box_max_shape[0] / image_height
        xs = width * box_max_shape[1] / image_width
        
        
        # box centre mapped with respect to full image
        _, H, W = self.feature_space_dim
        #print(H)
        h = torch.linspace(0.5, H-0.5, steps=H).to(self.device)
        w = torch.linspace(0.5, W-0.5, steps=W).to(self.device)
        
        yt = (self.pixels_per_cell[0] / image_height) * (cell_y + h[None, :, None, None])
        xt = (self.pixels_per_cell[1] / image_width) * (cell_x + w[None, None, :, None])
        
        z_where = torch.cat([xt, yt, xs, ys], dim=-1)
        return z_where, mean, std

    def _build_depth(self, depth_latent):
        depth_mean, depth_std = latent_to_mean_std(depth_latent)
        depth_logits = self._sample_z(depth_mean, depth_std, 'depth')
        z_depth = 4 * clamped_sigmoid(depth_logits)
        return z_depth
    
        
    def _build_what(self, what_latent):
        
        what_mean, what_std = latent_to_mean_std(what_latent)
        z_what = self._sample_z(what_mean, what_std, 'what')
    
        return z_what
        
        
    def _sample_z(self, mean, var, name):
        if self.mode == 'infer':
            return mean
            
        self.dist[name] = Normal(loc=mean, scale=var)
        return self.dist[name].rsample()
        
    def random_shuffle_var(self, objects, obj_prob, z_depth, z_where, x , output_image):
       num_color_channels = cfg.input_image_shape[0]
    
       # check if images in a batch is similar
       batch_size = objects.size(0)
       #print(batch_size)
       num_rows = min(5, batch_size)
       #print(num_rows)
       objects = objects.permute(0,2,3,1).contiguous()
       objects = debug_tools.torch2npy(objects, reshape=True)
       obj_prob = debug_tools.torch2npy(obj_prob, reshape=True)
       z_depth = debug_tools.torch2npy(z_depth, reshape=True)
       bounding_box = debug_tools.torch2npy(z_where, reshape=True)
       
       for i in range(16):
         #print(i)
         cat_obj_vec = objects[i, ...]
         # Bounding Box
         bbox = bounding_box[i, ...] * cfg.input_image_shape[-2] # image size
         presence = obj_prob[i, ...]
         #print(bbox)
         #print(presence)
       
    
    def max_deviation(self, l, a):
        
        sumof=0.0
        for i in range(len(l)):
           sumof+= ((a-l[i])**2)
        
        sdev = sumof/len(l)
        return math.sqrt(sdev)
        
        
    def torch2npy(self, t:torch.Tensor, reshape=False):
     shape = t.shape[1:]
     #print("t.shape = {}".format(t.shape))
     grid_size = int(math.sqrt(t.shape[0] / cfg.batch_size) + 1e-6)
     if reshape:
        return t.cpu().view(cfg.batch_size, grid_size, grid_size, *shape).detach().squeeze().numpy()
     return t.cpu().detach().numpy()


    def _render(self, z_what, z_where, z_depth, obj_prob, x):
        '''
        decoder + renderer function. combines the latent vars & bbox, feed into the decoder for each object and then
        '''
        _, H, W = self.feature_space_dim
        px = cfg.object_shape[0]
        
        z_where = z_where.view(-1, 4)
        z_depth = z_depth.view(-1, 1, 1)
        obj_prob = obj_prob.view(-1, 1, 1)
        object_decoder_in = z_what.view(-1, cfg.n_what)
    
        # generate image
        # object_logits : [batch*H*W, chan + 1, px, px]
        object_logits = self.object_decoder(object_decoder_in)
        
        # object_logits scale + bias mask
        chan = cfg.input_image_shape[0]
        object_logits = object_logits.view(-1, chan + 1, px, px)
        
        # add a bias to alpha channel, a magic number
        object_logits[:, -1, :, :] += 5.0
        
        objects = clamped_sigmoid(object_logits, use_analytical=True)

        torch.set_printoptions(threshold=10_000)
        #print(objects[:, -1, :, :][115])
        
        objects_mask=[]

        # incorporate presence in alpha channel
        print(obj_prob.shape)
        print(objects.shape)
        objects[:, -1, :, :] *= obj_prob.expand_as(objects[:, -1, :, :])
        

        importance = objects[:, -1, :, :] * z_depth.expand_as(objects[:, -1, :, :])
        importance = torch.clamp(importance, min = 0.0001)

        # merge importance to objects
        # objects : [B*H*W, 5, px, px]
        importance = importance.view(-1, 1, px, px)
        objects = torch.cat([objects, importance], dim = 1)
        
        img_c, img_h, img_w, = (self.image_shape)
        
        # transformed_imgs : [B*H*W, 5, img_h, img_w]
        
        transformed_imgs = stn(objects, z_where, [img_h, img_w],  self.device, inverse=True)
        #print(H)
        transformed_objects = transformed_imgs.contiguous().view(-1, H*W, img_c + 2 , img_h, img_w)
        
        


        color_channels  = transformed_objects[:, :, :img_c, :, :]
        
        #torch.set_printoptions(threshold=128*256*128*1*3)        
        #print(color_channels.view(1,16,16,3,128,128))
        
        alpha = transformed_objects[:, :, img_c:img_c+1, :, :] # keep the empty dim

        importance = transformed_objects[:, :, img_c+1:img_c+2, :, :] + 1e-9


        img = alpha.expand_as(color_channels) * color_channels
        
        #print(transformed_objects[:, :, :img_c, :, :].size())
        #print(color_channels.size())
        #print(importance.size())        
        
        # normalize importance
        importance = importance / importance.sum(dim=1, keepdim=True)
        importance = importance.expand_as(img)
        
        weighted_grads_image = img * importance

        output_image = weighted_grads_image.sum(dim=1) # sum up along n_obj per image

        # fix numerical issue
        output_image = torch.clamp(output_image, min=0, max=1)

        if self.mode == 'train':
            # set_trace()
            # img : [batch, H*W, imgh, imgw]
            img = torch.mean(img, dim=2) # mean for channels
            
            # overlap loss
            virtual_loss = torch.mean(torch.sum(torch.sum(img, dim=1) - torch.max(img, dim=1)[0], dim = [1,2]))
            
            '''if self.global_step >= 0 and self.global_step % 100 == 0:
                debug_tools.plot_render_components(objects, obj_prob, z_depth, z_where, x, output_image, self.writer, self.global_step)'''
            
            return output_image, virtual_loss, objects_mask, objects
        #print("The output images is \n")
        #print(output_image)
        #self.random_shuffle_var(objects, obj_prob, z_depth, z_where, x , output_image)
        
        return output_image, self._update_z_where(objects, z_where)
    
    
     
        
    def _update_z_where(self, objects, z_where):
        # calc upd_z_where
        input_chan = cfg.input_image_shape[0]
        color_chan = objects[:, :input_chan, :, :]
        alpha_chan = objects[:, input_chan:input_chan+1, :, :]
        color_objects = alpha_chan.expand_as(color_chan) * color_chan
        
        objects_mean_chan = torch.mean(color_objects, dim = 1)
        
        objects_max_y, _ = torch.max(objects_mean_chan, dim = 1)
        objects_max_x, _ = torch.max(objects_mean_chan, dim = 2)
        
        # find first False and last False
        num_obj = z_where.size(0)
        px = cfg.object_shape[0]
        objects_real_y0 = (torch.ones(num_obj) * px).to(self.device).long()
        objects_real_y1 = torch.zeros(num_obj).to(self.device).long()
        objects_real_x0 = (torch.ones(num_obj) * px).to(self.device).long()
        objects_real_x1 = torch.zeros(num_obj).to(self.device).long()
        
        objects_y_pos = torch.nonzero(objects_max_y >= cfg.virtual_b, as_tuple = False)
        objects_x_pos = torch.nonzero(objects_max_x >= cfg.virtual_b, as_tuple = False)
        
        for i in range(objects_y_pos.size(0)):
            objects_real_y0[objects_y_pos[i, 0]] = torch.min(objects_real_y0[objects_y_pos[i, 0]], objects_y_pos[i, 1])
            objects_real_y1[objects_y_pos[i, 0]] = torch.max(objects_real_y1[objects_y_pos[i, 0]], objects_y_pos[i, 1])
        
        for i in range(objects_x_pos.size(0)):
            objects_real_x0[objects_x_pos[i, 0]] = torch.min(objects_real_x0[objects_x_pos[i, 0]], objects_x_pos[i, 1])
            objects_real_x1[objects_x_pos[i, 0]] = torch.max(objects_real_x1[objects_x_pos[i, 0]], objects_x_pos[i, 1])
            
        objects_real_y1[objects_real_y1 < objects_real_y0] = 0
        objects_real_x1[objects_real_x1 < objects_real_x0] = 0
        
        objects_real_y0 = objects_real_y0
        objects_real_x0 = objects_real_x0
        objects_real_y1 = objects_real_y1 + 1
        objects_real_x1 = objects_real_x1 + 1
        
        objects_real_y0 = objects_real_y0.float()
        objects_real_y1 = objects_real_y1.float()
        objects_real_x0 = objects_real_x0.float()
        objects_real_x1 = objects_real_x1.float()
        
        upd_z_where = z_where.clone()
        upd_z_where[:, 0] = z_where[:, 2] / px * (objects_real_y0 + objects_real_y1 - px) / 2 + z_where[:, 0]
        upd_z_where[:, 1] = z_where[:, 3] / px * (objects_real_x0 + objects_real_x1 - px) / 2 + z_where[:, 1]
        upd_z_where[:, 2] = z_where[:, 2] / px * (objects_real_y1 - objects_real_y0)
        upd_z_where[:, 3] = z_where[:, 3] / px * (objects_real_x1 - objects_real_x0)
        
        _, H, W = self.feature_space_dim
        upd_z_where = upd_z_where.view(-1, H, W, 4)
        y = upd_z_where.view(-1)
        
        #print("Updated Z_Where\n" +str(upd_z_where))
        
        return upd_z_where


    def _compute_KL(self, cls_logits, z_cls, prob, obj_prob):
        
        # obj_prob [batch, H, W] -> [batch, H, W, 1]
        _, H, W = self.feature_space_dim
        obj_prob = obj_prob.view(-1,H,W,1)
        
        # p(z^{what} | z^{cls}) ~ N(mu(z^cls), var(z^cls))
        
        mu_prior = self.what_mu_prior(z_cls)
        var_prior = F.softplus(self.what_var_prior(z_cls))
        
        if self.global_step <= cfg.freezing_cat_iters:
            mu_prior = mu_prior.detach()
            var_prior = var_prior.detach()
            
        self.kl_priors['what'] = Normal(loc=mu_prior, scale=var_prior)
        
        
        KL = {}
        # For all latent distributions
        for dist_name in self.dist.keys():
           
            dist = self.dist[dist_name]
            prior = self.kl_priors[dist_name]
            
            kl_div = kl_divergence(dist, prior)
            
            masked = obj_prob * kl_div
            KL[dist_name] = masked

        pres_kl = obj_prob * (torch.log(obj_prob + 1e-9) - self.log_prior_obj) \
                  + (1 - obj_prob) * (torch.log(1 - obj_prob + 1e-9) - np.log(1 - self.prior_obj + 1e-9))                  
        log_cls = F.log_softmax(cls_logits, dim = -1)
        cls_kl = torch.sum(prob * log_cls, dim = -1, keepdim = True) + np.log(cfg.num_classes)
        
        KL['pres'] = pres_kl
        
        KL['cls'] = obj_prob * cls_kl
        
        return KL
        
    def _build_loss(self, x, recon_x, kl, virtual_loss, max_mask_prob, objects, obj_prob):
    
        print('============ Losses =============')
        # Reconstruction loss

        recon_loss = torch.mean(torch.sum(F.binary_cross_entropy(recon_x, x, reduction='none'), dim=[1,2,3]))
        
        #fake_bbox loss
        obj_prob = obj_prob.view(-1, 1, 1)
        objects_mask=[]
        for i in range(1,256):
           objects_mask.append(torch.square(torch.tensor(1.0, requires_grad=True)-torch.max(objects[:, -1, :, :][i]))*torch.max(obj_prob.expand_as(objects[:, -1, :, :])[i]).item())
           
        fbbox_loss = torch.tensor(max(objects_mask), requires_grad=True)
        self.writer.add_scalar('losses/reconst', recon_loss, self.global_step)
        

        
        print('Reconstruction loss:', '{:.4f}'.format(recon_loss.item()))

        # KL loss with Beta factor
        kl_loss = 0
        for name, z_kl in kl.items():
            kl_mean = torch.mean(torch.sum(z_kl, dim=[1,2,3])) # batch mean
            
            kl_loss += cfg.alpha[name] * kl_mean
            
            print('KL_%s_loss:' % name, '{:.4f}'.format(kl_mean.item()))
            self.writer.add_scalar('losses/KL_{}'.format(name), kl_mean, self.global_step)
            
        # Category Similarity (can it be a virtual loss ?)
        ''' print('Beta Loss is: ', betaloss)
        self.writer.add_scalar('losses/BetaLoss', betaloss, self.global_step)'''
        w = F.normalize(self.what_mu_prior.weight, dim = 0)
        mat = torch.matmul(torch.transpose(w, 0, 1), w)
        # print('Mat : {}'.format(mat))
        cat_sim = torch.sum(mat)
        print('Cat similarity:', '{:.4f}'.format(cat_sim.item()))
        self.writer.add_scalar('losses/Cat_similarity', cat_sim, self.global_step)

        # Virtual Loss
        print('Overlap loss:', '{:.4f}'.format(virtual_loss.item()))
        self.writer.add_scalar('losses/Overlap_loss', virtual_loss, self.global_step)
        
        

        loss = cfg.alpha['recon'] * recon_loss + kl_loss + fbbox_loss
        
        if self.global_step <= cfg.virtual_loss_step:
            loss += cfg.alpha['virtual'] * virtual_loss
            
        print('Fake Bounding Box Loss:',fbbox_loss)
        self.writer.add_scalar('losses/fake_bbox', fbbox_loss, self.global_step)
        print('\n ===> total loss:', '{:.4f}'.format(loss.item()))
        self.writer.add_scalar('losses/total', loss, self.global_step)
        
        print("THIS IS THE LOSS IM TALKING ABOUT")
        print(loss)
        #print(loss-fbbox_loss)

        return loss

    def generate(self, x, alter_type=None):
        # --- generate new images ---
        # alter_type : 'category', 'style', 'position'
        
        self.mode = 'infer'
        
        # feat : [batch, C, H, W]
        feat = self.backbone(x)
        
        # cls_logits : [batch, H, W, num_classes + 1]
        pres_logits = self.heat_map_head(feat).permute(0,2,3,1).contiguous()
        
        # where_latent : [batch, H, W, 8]
        where_latent = self.where_head(feat).permute(0,2,3,1).contiguous()
        
        # depth_latent : [batch, H, W, 2]
        depth_latent = self.depth_head(feat).permute(0,2,3,1).contiguous()
        
        # z_where : [batch, H, W, 4]
        z_where = self._build_where(where_latent)
        
        # z_depth : [batch, H, W, 1]
        z_depth = self._build_depth(depth_latent)
        
        # z_what : [batch, H, W, N_WHAT]
        # what_latent, cls_logits = self._encode_obj(x, z_where)
        
        _, H, W = self.feature_space_dim
        input_glimpses = stn(x.repeat_interleave(H*W, dim = 0), z_where.view(-1, 4), cfg.object_shape, self.device)
        input_glimpses = input_glimpses.flatten(start_dim = 1)
        cls_logits = self.object_encoder_cat(input_glimpses).view(-1,H,W,cfg.num_classes)
        
        # z_cls : [batch, H, W, num_classes] (gumbel softmax)
        # prob : [batch, H, W, num_classes] (softmax)
        # obj_prob : [batch, H, W]
        z_cls, prob, obj_prob = self._build_cls(pres_logits, cls_logits)
        
        concat = torch.cat([input_glimpses, z_cls.view(-1,cfg.num_classes)], dim = -1)
        what_latent = self.object_encoder_what(concat).view(-1,H,W,2*cfg.n_what)
        
        z_what = self._build_what(what_latent)
        
        w_mu = self.what_mu_prior.weight.detach() + self.what_mu_prior.bias.detach().view(-1,1)
        w_mu = w_mu.transpose(0, 1)
        w_var = F.softplus(self.what_var_prior.weight.detach() + self.what_var_prior.bias.detach().view(-1, 1))
        w_var = w_var.transpose(0, 1)
        
        
        _, cat = torch.max(z_cls, dim = -1)
        
        B = x.size(0)
        
        if alter_type == 'category':
            print(obj_prob[0, obj_prob[0] >= 0.5])
            np.random.seed(10)
            g_cat = np.random.randint(cfg.num_classes, size = B)
            g_cat = torch.from_numpy(g_cat).to(self.device).view(B, 1, 1).expand_as(cat)
            z_what[obj_prob >= 0.5] = z_what[obj_prob >= 0.5] - w_mu[cat[obj_prob >= 0.5]] + w_mu[g_cat[obj_prob >= 0.5]]
        elif alter_type == 'style':
            dist = Normal(0, 1)
            z_local = dist.sample(z_what.size()).to(self.device)
            z_what[obj_prob >= 0.5] = w_mu[cat[obj_prob >= 0.5]] + z_local[obj_prob >= 0.5]
        elif alter_type == 'position':
            np.random.seed(4)
            
            for i in range(B):
                idx = np.random.permutation(H*W)
                idx = torch.from_numpy(idx).to(self.device)
                z_what[i] = z_what[i].detach().view(H*W, -1)[idx].view(H, W, -1)
                where_latent[i] = where_latent[i].detach().view(H*W, -1)[idx].view(H, W, -1)
                z_depth[i] = z_depth[i].detach().view(H*W, -1)[idx].view(H, W, -1)
                obj_prob[i] = obj_prob[i].detach().view(H*W)[idx].view(H, W)
                
            z_where = self._build_where(where_latent)
        else:
            assert False
            
        recon_x, _ = self._render(z_what, z_where, z_depth, obj_prob, x)
            
        return recon_x

