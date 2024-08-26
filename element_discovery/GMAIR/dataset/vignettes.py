import os
import h5py
import numpy as np
import cv2

import torch

from gmair.config import config as cfg

class Vignettes(torch.utils.data.Dataset):
    def __init__(self, root, anno_file):
        super().__init__()
        self.root = root
        self.info = anno_file
        self.imageFileNames = os.listdir(root)
        self.annoFileNames = os.listdir(anno_file)
        self.imageFileNames.sort()
        self.annoFileNames.sort()
        self.train_images = []
        self.anno = {}
        self.count = {}
        
        for f in self.imageFileNames:
            self.train_images.append(f)
        i = 0
        for f in self.annoFileNames:
            with open(os.path.join(self.info, f), "r") as fl:
                lines = fl.readlines()
                bbox = []
                for line in lines:
                    line = line.strip('\n')
                    datas = line.split(',')
                    if datas.__len__() == 1:
                        self.count[i] = (int(datas[0]))
                    else:
                        c=1
                        x, y, w, h = datas
                        bbox.append([float(x) - float(w)/2, float(y) - float(h)/2, 
                            float(w), float(h), float(c)])
                bbox = np.array(bbox)
                
                if self.count[i] > 100:
                    self.count[i] = 100
                    self.anno[i] = bbox[:100, :]
                else:
                    if self.count[i] == 0:
                        self.anno[i] = -np.ones((100, 5),dtype=int)
                        #print(f)
                    else:
                        self.anno[i] = np.row_stack((bbox, -np.ones((100 - self.count[i], 5),dtype=int)))
                i += 1    
                 

    def __getitem__(self, index):
        img_info = self.train_images[index]
        
        img_path = os.path.join(self.root, img_info)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_CUBIC)
        # image = image[..., None]
        image = np.moveaxis(image, -1, 0)
        image = image.astype(np.float32)
        image /= 255.0
        '''try:
         bbox = self.anno[index]
         count = self.count[index]
         print(bbox)
         print(count)
        except:
         bbox=[]
         count=[]'''
        return image
    

    def __len__(self):
        return len(self.train_images)
        
