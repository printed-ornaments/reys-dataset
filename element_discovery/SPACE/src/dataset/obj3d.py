from torch.utils.data import Dataset
import os.path as osp
from torchvision import transforms
from skimage import io
import os
import numpy as np
import torch
from PIL import Image, ImageFile
import torchvision.transforms.functional as F
from torchvision.io import read_image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Obj3D(Dataset):
    def __init__(self, root, mode):
        # checkpointdir = os.checkpointdir.join(root, mode)
        assert mode in ['train', 'val', 'test']
        self.root = root
        self.mode = mode
        assert os.path.exists(root), 'Path {} does not exist'.format(root)
        
        
        self.img_paths = []
        img_dir = os.path.join(self.root, 'val')
        for file in os.scandir(img_dir):
            img_path = file.path
            if 'png' in img_path or 'jpg' in img_path:
                self.img_paths.append(img_path)
        
        get_index = lambda x: int((os.path.basename(x).split('image_')[1]).split('.')[0])
        self.img_paths.sort(key=get_index)
        
    @property
    def bb_path(self):
        path = osp.join(self.root, 'bb')
        assert osp.exists(path), f'Bounding box path {path} does not exist.'
        return path
    
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = io.imread(img_path)[:, :, :3]

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize([128,128]),
            transforms.ToTensor(),
        ])
        img = transform(img)

        return img
    
    def __len__(self):
        return len(self.img_paths)
    
