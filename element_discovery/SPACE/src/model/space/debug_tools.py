''' Modified based on https://github.com/yonkshi/SPAIR_pytorch '''

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from cycler import cycler
from matplotlib.collections import PatchCollection
import torch
import numpy as np
import time
import math
import cv2
import torchvision.transforms as T

#from gmair.utils.coolname import generate_slug
from PIL import Image
import config as cfg
transform = T.ToPILImage()

def plot_torch_image_in_pyplot( out:torch.Tensor, inp:torch.Tensor = None, batch_n=0):
    ''' For visualizing torch images in matplotlib '''
    torch_img = out[batch_n, ...]
    np_img = torch_img.detach().numpy()
    np_img = np.moveaxis(np_img,[0,1,2], [2,0,1]) # [C, H, W] -> [H, W, C]
    np_img = np.squeeze(np_img)
    plt.imshow(np_img)
    plt.title('out_image')
    plt.show()

    if inp is not None:
        torch_img = inp[batch_n, ...]
        np_img = torch_img.detach().numpy()
        np_img = np.moveaxis(np_img,[0,1,2], [2,0,1]) # [C, H, W] -> [H, W, C]
        plt.imshow(np_img)
        plt.show()

def benchmark_init():
    global BENCHMARK_INIT_TIME
    BENCHMARK_INIT_TIME = time.time()

def benchmark(name='', print_benchmark=True):
    global BENCHMARK_INIT_TIME
    now = time.time()
    diff = now - BENCHMARK_INIT_TIME
    BENCHMARK_INIT_TIME = now
    if print_benchmark: print('{}: {:.4f} '.format(name, diff))
    return diff

def torch2npy(t:torch.Tensor, reshape=False):
    '''
    Converts a torch graph node tensor (cuda or cpu) to numpy array
    :param t:
    :return:
    '''
    shape = t.shape[1:]
    #print("t.shape = {}".format(t.shape))
    grid_size = int(math.sqrt(t.shape[0] / cfg.batch_size) + 1e-6)
    if reshape:
        return t.cpu().view(cfg.batch_size, grid_size, grid_size, *shape).detach().squeeze().numpy()
    return t.cpu().detach().numpy()

def plot_render_components(obj_vec, obj_prob, depth, bounding_box, input_image, output_image, writer, step):

    ''' Plots each component prior to rendering '''
    num_color_channels = cfg.input_image_shape[0]
    
    # check if images in a batch is similar
    batch_size = obj_vec.size(0)
    num_rows = min(5, batch_size)
    num_cols = 7 + 10
    
    gs = gridspec.GridSpec(num_rows, num_cols)
    fig = plt.figure(figsize = (45, 20))
    fig.tight_layout()
    plt.tight_layout()
    
    obj_vec = obj_vec.permute(0,2,3,1).contiguous()
    obj_vec = torch2npy(obj_vec, reshape=True)
    obj_prob = torch2npy(obj_prob, reshape=True)
    depth = torch2npy(depth, reshape=True)
    bounding_box = torch2npy(bounding_box, reshape=True)
    
    for i in range(num_rows):
        cat_obj_vec = obj_vec[i, ...]
        cat_obj_vec = np.concatenate(cat_obj_vec, axis=-3) # concat h
        cat_obj_vec = np.concatenate(cat_obj_vec, axis=-2) # concat w
        
        in_img = input_image[i,...].permute(1,2,0).cpu().detach().squeeze().numpy()
        out_img = output_image[i,...].permute(1,2,0).cpu().detach().squeeze().numpy()
        
        # Bounding Box
        bbox = bounding_box[i, ...] * cfg.input_image_shape[-2] # image size
        presence = obj_prob[i, ...]
        _plot_bounding_boxes('input image', bbox, in_img, presence,  gs[i, 0], fig)
        _plot_bounding_boxes('output image', bbox, out_img, presence, gs[i, 1], fig)
        
        # Attr image
        obj = cat_obj_vec[..., :num_color_channels]
        #_plot_image('rendered_obj', obj, gs[i, 2], fig)
    
        # Alpha Channel (heatmap)
        alpha = cat_obj_vec[..., num_color_channels]
        _plot_heatmap('alpha', alpha, gs[i, 3], fig, cmap='spring')
    
        # Importance (heatmap)
        impo = cat_obj_vec[..., num_color_channels+1]
        _plot_heatmap('importance', impo, gs[i, 4], fig, cmap='summer')
    
        # depth (heatmap)
        dep = depth[i,...]
        _plot_heatmap('depth', dep, gs[i, 5], fig, cmap='autumn')
        
        # Presence (heatmap)
        _plot_heatmap('presence', presence, gs[i, 6], fig, cmap='winter')
        
        H, W = presence.shape
        index = np.argsort(presence.reshape(-1))[::-1]
        for j in range(num_cols - 7):
            obj_j = obj_vec[i, index[j] // W, index[j] % W, :, :, :num_color_channels]
            alpha_j = obj_vec[i, index[j] // W, index[j] % W, :, :, num_color_channels:num_color_channels+1]
            obj_j = obj_j * alpha_j
            #_plot_image('rendered_obj_{}'.format(j), obj_j, gs[i, j + 7], fig)
    
    
    writer.add_figure('renderer_analysis'.format(i), fig, step)
    
def plot_infer_render_components(input_image, obj_prob, pred_cls, pred_bbox, output_image, step):

    ''' Plots each component prior to rendering '''
    input_image_copy = input_image
    output_image_copy = output_image

    input_image = input_image[0,...].permute(1,2,0).cpu().detach().squeeze().numpy()
    output_image = output_image[0,...].permute(1,2,0).cpu().detach().squeeze().numpy()
    
    # [x, y, w, h, c, conf]
    '''gt_bbox = gt_bbox[0]
    gt_bbox = gt_bbox[gt_bbox[:, 0] >= 0].cpu().detach().numpy()
    gt_bbox = np.concatenate([gt_bbox, np.ones([gt_bbox.shape[0], 1], dtype = np.float)], axis = 1)
    print('ground truth shqpe')
    print(gt_bbox.shape)'''
    
    ori_obj_prob = obj_prob[0].cpu().detach().numpy()
    obj_prob = ori_obj_prob.reshape(-1)
    
    pred_conf, pred_cls = torch.max(pred_cls[0], dim = -1)
    pred_conf = pred_conf.view(-1, 1).cpu().detach().numpy()[obj_prob >= 0.5]
    pred_cls = pred_cls.view(-1, 1).cpu().detach().numpy()[obj_prob >= 0.5]
    
    pred_bbox = pred_bbox[0].view(-1, 4).cpu().detach().numpy()[obj_prob >= 0.5]
    pred_bbox *= 128
    pred_bbox[:, 0] -= pred_bbox[:, 2] / 2
    pred_bbox[:, 1] -= pred_bbox[:, 3] / 2
    print('bbox shqpe')
    print(pred_bbox.shape)
    print(pred_conf.shape)
    # sprint(pred_cls.shape)
    pred_bbox = np.concatenate([pred_bbox, pred_cls, pred_conf], axis = 1)
    
    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize = (15,10))
    fig.tight_layout()
    plt.tight_layout()


    # Bounding Box
    #_plot_bounding_boxes_w_c('x', gt_bbox, input_image, gs[0, 0], fig, input_image_copy, step, 'groundtruth')

    _plot_bounding_boxes_w_c('x_hat', pred_bbox, output_image, gs[0, 1], fig, output_image_copy, step, 'output')
    
    # Presence (heatmap)
    _plot_heatmap('obj_prob', ori_obj_prob, gs[0, 2], fig, cmap='winter')

    
    
        
def plot_cropped_input_images(cropped_input_images, writer, step):
    input_imgs = cropped_input_images.permute(0,4,5, 2,3,1,).cpu().squeeze().detach().numpy()
    # np.swapaxes(input_imgs, )
    input_img = input_imgs[0,...]
    H = input_img.shape[0]
    W = input_img.shape[1]

    # adding border to cropped images
    px_h = px_w = input_img.shape[-1] + 2
    input_img_with_border = np.ones([H, W, px_h, px_w])
    input_img_with_border[..., 1:-1, 1:-1] = input_img


    img = np.concatenate(input_img_with_border, axis=-2) # concat h
    img = np.concatenate(img, axis=-1) # concat w

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(img, cmap='gray' , vmin=0, vmax=1)

    if cfg.IS_LOCAL:
        plt.show()
    else:
        writer.add_figure('debug_cropped_input_images', fig, step)

def plot_objet_attr_latent_representation(z_attr, writer, step, title='z_attr/heatmap'):
    z_attr = z_attr[0, ...]
    z_attr = torch2npy(z_attr)

    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize = (7,2.5))
    fig.tight_layout()
    plt.tight_layout()

    z_attr_max = z_attr.max(axis=0)
    _plot_heatmap('Max', z_attr_max, gs[0, 0], fig, cmap='spring')

    z_attr_mean = z_attr.mean(axis=0)
    _plot_heatmap('Mean', z_attr_mean, gs[0, 1], fig, cmap='spring')

    z_attr_min = z_attr.min(axis=0)
    _plot_heatmap('Min', z_attr_min, gs[0, 2], fig, cmap='spring')

    if cfg.IS_LOCAL:
        plt.show()
    else:
        writer.add_figure(title, fig, step)


def _plot_heatmap(title, data, gridspec, fig, cmap, vmin=0, vmax=1):
    ax = fig.add_subplot(gridspec)
    im = ax.imshow(data, cmap=cmap)
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

def _plot_image(title, data, gridspec, fig):
    ax = fig.add_subplot(gridspec)
    print(gridspec)
    im = ax.imshow(data, cmap='gray' , vmin=0, vmax=1) # Specific to the MNIST dataset
    # Disable axis
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)

def _plot_bounding_boxes(title, bbox, original_image, z_pres, gridspec, fig):

    ax = fig.add_subplot(gridspec)
    ax.imshow(original_image, cmap='gray', vmin=0, vmax=1)
    #ptchs = []
    H, W, _ = bbox.shape
    for i in range(H):
        for j in range(W):
            x, y, w, h = bbox[i, j]
            if z_pres[i, j] < 0.5:
                continue
            pres = np.clip(z_pres[i, j], 0.5, 1)
            # border_color = (1,0,0,pres) if pres > 0.5 else (0,0,1, pres)# red box if > 0.5, otherwise blue
            border_color = (1,0,0,pres)
            # Green box: ground truth, red box: inferrence, blue box: disabled inferrence

            x -= w/2
            y -= h/2
            patch = patches.Rectangle([x,y], w, h, facecolor='none', edgecolor=border_color, linewidth=1)
            ax.add_patch(patch)

    # ax.add_collection(PatchCollection(ptchs, facecolors='none', edgecolors='r', linewidths=1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def _plot_bounding_boxes_w_c(title, bbox, original_image, gridspec, fig, images, step, mode):
    print('Entered w c module')
    ax = fig.add_subplot(gridspec)
    ax.imshow(original_image, cmap='gray', vmin=0, vmax=1)
    fin_img = transform(images[0])
    #rand_name = generate_slug()
    st = '../results/examples_SPAGMACE/gmairx/'+str(step)+'_new_full_bbox'+'.png'
    fin_img.save(st)
    #plt.figure()
    #plt.imshow(fin_img)
    #plt.show()
    fin_img =cv2.imread(st)
    
    num, _ = bbox.shape
    for i in range(num):
        x, y, w, h, c, conf = bbox[i]
        if mode == 'output':
           fin_img = cv2.rectangle(fin_img,(int(x)-int(float(w)/6),int(y)),(int(x)+int(float(w)),int(y)+int(float(h))),(255,0,0),1)
        #plt.figure()
        #plt.imshow(fin_img)
        #plt.show()
        aaa = np.array(fin_img)
        abc = Image.fromarray(aaa)
        abc.save(st)

        border_color = (1,0,0,conf)
        
        patch = patches.Rectangle([x, y], w, h, facecolor='none', edgecolor=border_color, linewidth=1)
        ax.add_patch(patch)
        
        patch = patches.Rectangle([x, y-3], w, 3, facecolor=(1,1,1,1))
        ax.add_patch(patch)
        
        ax.text(x, y, '{}'.format(int(c + 0.01)), 
            fontsize=6,
            # style='italic',
            # bbox={'facecolor': 'white', 'alpha': conf, 'pad': (0,0,2,0)}
            )
        

    # ax.add_collection(PatchCollection(ptchs, facecolors='none', edgecolors='r', linewidths=1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)


def decoder_output_grad_hook(grad, writer, step):
    if step % 10 != 0:
        return
    obj_px = cfg.OBJECT_SHAPE[0]
    grad = grad.view(cfg.batch_size, GRID_SIZE, GRID_SIZE, obj_px, obj_px, 2).cpu().detach().numpy()
    grad = grad[0, ...]
    obj_vec = np.concatenate(grad, axis=-3) # concat h
    obj_vec = np.concatenate(obj_vec, axis=-2) # concat w
    img = obj_vec[...,0]

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(img, vmin=-1e-4, vmax=1e-4)
    plt.title('gradient of decoder')
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)

    if cfg.IS_LOCAL:
        plt.show()
        print('')
    else:
        writer.add_figure('grad_visualization/decoder_out', fig, step)

def z_attr_grad_hook(grad, writer, step):
    if step % 50 != 0:
        return
    # grad = grad.view(2, cfg.N_ATTRIBUTES, 11, 11, 2).squeeze().detach().numpy()
    z_attr_grad = torch2npy(grad[0, ...])

    gs = gridspec.GridSpec(1, 3)
    fig = plt.figure(figsize = (7,2.5))
    fig.tight_layout()
    plt.tight_layout()

    z_attr_max = z_attr_grad.max(axis=0)
    _plot_heatmap('Max', z_attr_max, gs[0, 0], fig, cmap='spring')

    z_attr_mean = z_attr_grad.mean(axis=0)
    _plot_heatmap('Mean', z_attr_mean, gs[0, 1], fig, cmap='spring')

    z_attr_min = z_attr_grad.min(axis=0)
    _plot_heatmap('Min', z_attr_min, gs[0, 2], fig, cmap='spring')

    if cfg.IS_LOCAL:
        plt.show()
        print('')
    else:
        writer.add_figure('grad_visualization/z_attr', fig, step)

def nan_hunter(name, **kwargs):

    nan_detected = False
    tensors = {}
    non_tensors = {}
    for name, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            tensors[name] = value
            if torch.isnan(value).sum() > 0:
                nan_detected = True
        else:
            non_tensors[name] = value

    if not nan_detected: return


    print('======== NAN DETECTED in %s =======' % name)

    for name, value in non_tensors.items():
        print(name, ":", value)

    for name, tensor in tensors.items():
        print(name, ':\n', tensor)

    print('======== END OF NAN DETECTED =======')

    raise AssertionError('NAN Detected by Nan detector')




