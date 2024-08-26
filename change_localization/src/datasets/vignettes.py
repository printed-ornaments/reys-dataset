import torchvision.transforms.functional
from src.utils.paths import DATA_PATH
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


class Vignettes:
    def __init__(self, glyph):
        super(Vignettes, self).__init__()
        self.glyph = glyph
        data_path = os.path.join(DATA_PATH, glyph)
        samples = []
        indices = {}
        count = 0

        for cat in ['normal', 'changed', 'unchanged']:
            all_sub_dirs = os.listdir(os.path.join(data_path, cat))
            for item in all_sub_dirs:
                if item[-3:] != 'txt':
                    if item[-3:] == 'tif':
                        self.label = torch.tensor(np.array(Image.open(os.path.join(data_path, cat, item)))/255).permute(2, 0, 1).sum(0).int()
                    else:
                        samples.append(torch.tensor(plt.imread(os.path.join(data_path, cat, item)) / 255).permute(2, 0, 1).float())
                        indices[count] = cat
                        count += 1

        height, width = self.label.shape
        samples = resize_to_shape(samples, height, width)
        samples_dict = {}
        for cat in ['normal', 'changed', 'unchanged']:
            curr_list = []
            for count in range(len(samples)):
                if indices[count] == cat:
                    curr_list.append(samples[count])
            try:
                samples_dict[cat] = torch.stack(curr_list, dim=0)
            except:
                samples_dict[cat] = torch.tensor([])
        self.samples = samples_dict


def resize_to_shape(image_tensors, height, width):
    resized_images = []
    for image in image_tensors:
        resized_image = torchvision.transforms.functional.resize(image, [height, width])
        resized_images.append(resized_image)
    return resized_images