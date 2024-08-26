import argparse
import os
import numpy as np
import cv2
import datetime
import random
from ipdb import set_trace

import torch
from torch import autograd
from torch import nn, optim
from tensorboardX import SummaryWriter

from gmair.models.model import gmair
from gmair.config import config as cfg
from gmair.dataset.fruit2d import FruitDataset
from gmair.dataset.multi_mnist import SimpleScatteredMNISTDataset
from gmair.utils import debug_tools
from gmair.test import metric, cluster_metric
from PIL import Image
from itertools import chain
from math import ceil
from PIL import Image
parser = argparse.ArgumentParser()
parser.add_argument('--test_num_iters', type=int, default=1000, help='test number of iterations')

args = parser.parse_args()

random.seed(3)
torch.manual_seed(3)

log_dir = os.path.join(
        cfg.log_dir,
        'test',
        datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
log_dir = os.path.abspath(log_dir)
writer = SummaryWriter(log_dir)
print('log path : {}'.format(log_dir))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test():
    image_shape = cfg.input_image_shape
    
    if cfg.dataset == 'multi_mnist':
        data = SimpleScatteredMNISTDataset(os.path.join(cfg.dataset_dir, 'scattered_mnist_128x128_obj14x14.hdf5'), mode = 'test')
    elif cfg.dataset == 'fruit2d':
        data = FruitDataset('./GMAIR-pytorch/gmair/data/testactual/','./GMAIR-pytorch/gmair/data/testbb/')
    else:
        print('No implementation for {}'.format(cfg.dataset))
        exit(0)
        

    gmair_net = gmair(image_shape, writer, device).to(device)

    #gmair_net.load_state_dict(torch.load("./logs/hehelabel/2022_09_13__17_28_25/checkpoints/step_16000.pkl"))
    gmair_net.load_state_dict(torch.load("./logs/hehelabel/2023_03_07__12_00_26/checkpoints/step_2600.pkl"))
    #gmair_net.load_state_dict(torch.load("./logs/hehelabel/2022_10_06__14_59_57/checkpoints/step_7000.pkl"))
    # os.path.join(cfg.test_model_path, 'checkpoints', 'step_{}.pkl'.format(cfg.test_iter))))
    
    torch.manual_seed(10)
    dataloader = torch.utils.data.DataLoader(data,
                                       batch_size = cfg.test_batch_size,
                                       pin_memory = True,
                                       num_workers = cfg.num_workers,
                                       drop_last = False,
                                       shuffle = True
                                       )
                                       
    tot_iter = min(len(dataloader), args.test_num_iters)
    # print(len(dataloader))
  
    all_z_where = None
    all_z_cls = None
    all_obj_prob = None
    all_y_bbox = None
    all_y_obj_count = None
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= tot_iter:
            break
            
        if (batch_idx + 1) % 100 == 0:
            print('test {} / {}'.format(batch_idx + 1, tot_iter))
        
        x_image, y_bbox, y_obj_count = batch
        
        x_image = x_image.to(device)
        y_bbox = y_bbox.to(device)
        y_obj_count = y_obj_count.to(device)
        print(x_image.shape)

        
        with torch.no_grad():
            gmair_net.eval()
            out_img, z_cls, z_what, z_where, obj_prob = gmair_net(x_image, mode = 'infer')
            
        all_z_where = z_where if all_z_where is None else torch.cat([all_z_where, z_where], dim = 0)
        all_z_cls = z_cls if all_z_cls is None else torch.cat([all_z_cls, z_cls], dim = 0)
        all_obj_prob = obj_prob if all_obj_prob is None else torch.cat([all_obj_prob, obj_prob], dim = 0)
        all_y_bbox = y_bbox if all_y_bbox is None else torch.cat([all_y_bbox, y_bbox], dim = 0)
        all_y_obj_count = y_obj_count if all_y_obj_count is None else torch.cat([all_y_obj_count, y_obj_count], dim = 0)
        print("ALL THE CLASSES ARE")
        print(str((torch.unique(all_z_cls)).size))    
        debug_tools.plot_infer_render_components(x_image, y_bbox, obj_prob, z_cls, z_where, out_img, writer, batch_idx)
    print(all_obj_prob.size())
    k=0
    
    meanAP = metric.mAP(all_z_where, all_obj_prob, all_y_bbox[:, :, :4], all_y_obj_count)
    print('Bbox Average Precision : ', meanAP)
  
    
    acc, nmi = cluster_metric.test_cluster(all_z_where, all_z_cls, all_obj_prob, all_y_bbox)
    print('Cluster Accuracy : ', acc)
    print('Cluster NMI : ', nmi)



        

if __name__ == '__main__':
    test()


