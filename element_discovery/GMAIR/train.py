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
from gmair.dataset.vignettes import Vignettes
#from gmair.dataset.multi_mnist import SimpleScatteredMNISTDataset
from gmair.utils import debug_tools
#from gmair.test import metric, cluster_metric

# args = parser.parse_args()
print('Entered Training Module')
random.seed(3)
torch.manual_seed(3)
np.random.seed(3)

log_dir = os.path.join(
        cfg.log_dir,
        datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
log_dir = os.path.abspath(log_dir)
writer = SummaryWriter(log_dir)
print('log path : {}'.format(log_dir))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train():
    load_model_path = '/home_expes/cs103329/cs103329/GMAIR-pytorch-2/logs/hehelabel/2024_02_04__19_07_54/checkpoints/step_27500.pkl'
    #load_model_path=None
    start_iter = 27500
    
    image_shape = cfg.input_image_shape
    
    if cfg.dataset == 'multi_mnist':
        data = SimpleScatteredMNISTDataset('./GMAIR-pytorch/gmair/data/multi_mnist/scattered_mnist_128x128_obj14x14.hdf5')

    elif cfg.dataset == 'fruit2d':
        data = Vignettes('/home_expes/cs103329/cs103329/GMAIR-pytorch-2/gmair/data/val','/home_expes/cs103329/cs103329/check/Vignette_Scattered_Data/bb')
        #test_data =  Vignettes('./GMAIR-pytorch/gmair/data/em/img/','./GMAIR-pytorch/gmair/data/em/bb')
    else:
        print('No implementation for {}'.format(cfg.dataset))
        exit(0)

    gmair_net = gmair(image_shape, writer, device).to(device)
    
    if load_model_path is not None:
        print('loading checkpoint')
        gmair_net.load_state_dict(torch.load(load_model_path))
            
    encoder_params = list(map(id, gmair_net.object_encoder_what.parameters()))
    decoder_params = list(map(id, gmair_net.object_decoder.parameters()))
    encoder_cat_params = list(map(id, gmair_net.object_encoder_cat.parameters()))
    
    pre_params = encoder_params + decoder_params + encoder_cat_params
    
    other_params = filter(lambda p: id(p) not in pre_params, gmair_net.parameters())
    
    params = [
      {"params": gmair_net.object_encoder_what.parameters(), "lr": cfg.encoder_what_lr},
      {"params": gmair_net.object_decoder.parameters(), "lr": cfg.decoder_lr},
      {"params": gmair_net.object_encoder_cat.parameters(), "lr": cfg.encoder_cat_lr},
      {"params": other_params, "lr": cfg.lr},
    ]
    
    gmair_optim = optim.Adam(params, lr=cfg.lr)

    start_cluster_measure = False
    
    for epoch in range(3, 500):
        dataloader = torch.utils.data.DataLoader(data,
                                           batch_size = 4,
                                           pin_memory = True,
                                           num_workers = cfg.num_workers,
                                           drop_last = True,
                                           shuffle = True)

        print("epoch {}".format(epoch))
        for batch_idx, batch in enumerate(dataloader):
            iteration = epoch * len(dataloader) + batch_idx + start_iter
            
            x_image = batch
            
            x_image = x_image.to(device)
            #print(x_image.shape())
            
            
            '''y_bbox = y_bbox.to(device)
            y_obj_count = y_obj_count.to(device)'''
            print("epoch {}".format(epoch))

            print('Iteration', iteration)

            # with autograd.detect_anomaly():
            gmair_net.train()
            gmair_optim.zero_grad()
            loss, out_img = gmair_net(x_image, iteration)
            loss.backward() #retain_graph = True)
            gmair_optim.step()
             
            # logging stuff
            image_out = out_img[0]
            image_in = x_image[0]
            combined_image = torch.cat([image_in, image_out], dim=2)
            writer.add_image('gmair reconstruction', combined_image, iteration)

            
            if iteration > -1 and iteration % 1 == 0:  # iteration > 1000 and
                with torch.no_grad():
                    gmair_net.eval()
                    out_img, z_cls, z_what, z_where, obj_prob = gmair_net(x_image, mode = 'infer')
                debug_tools.plot_infer_render_components(x_image, obj_prob, z_cls, z_where, out_img, writer, iteration)                
                
                #meanAP = metric.mAP(z_where, obj_prob, y_bbox[:, :, :4], y_obj_count)
                '''print('Bbox Average Precision : ', meanAP)
                writer.add_scalar('metric/bbox_average_precision', meanAP, iteration)'''
                
                '''if meanAP > 0.5:
                    start_cluster_measure = True
                    
                if start_cluster_measure:
                    #acc, nmi = cluster_metric.test_cluster(z_where, z_cls, obj_prob, y_bbox)
                    print('Cluster Accuracy : ', acc)
                    print('Cluster NMI : ', nmi)
                    writer.add_scalar('metric/cluster_accuracy', acc, iteration)
                    writer.add_scalar('metric/cluster_nmi', nmi, iteration)'''
            
            
            # Save model
            '''if iteration > 0 and iteration % 10 == 0:
                check_point_name = 'step_%d.pkl' % iteration
                cp_dir = os.path.join(log_dir, 'checkpoints')
                os.makedirs(cp_dir, exist_ok=True)
                save_path = os.path.join(log_dir, 'checkpoints', check_point_name)
                torch.save(gmair_net.state_dict(), save_path)'''
            print('=================\n\n')
            
            # torch.cuda.empty_cache()


if __name__ == '__main__':
    train()
