import torchvision
import numpy as np
from pathlib import Path


def gaussian_blur(img):
    img = torchvision.transforms.functional.gaussian_blur(img, kernel_size=(5, 9), sigma=(0.1, 3))
    return img


def color_jitter_(img, b, s, c, h, g):
    img = torchvision.transforms.functional.adjust_brightness(img, b)
    img = torchvision.transforms.functional.adjust_contrast(img, c)
    img = torchvision.transforms.functional.adjust_saturation(img, s)
    img = torchvision.transforms.functional.adjust_hue(img, h)
    img = torchvision.transforms.functional.adjust_gamma(img, g)
    return img


def color_jitter(data, b=0.4, s=0.5, c_min=0.2, c_max=1.1, h=0.05):
    brightness_factor = (2 * np.random.rand() - 1) * b + 1
    saturation_factor = (2 * np.random.rand() - 1) * s + 1
    contrast_factor = np.random.rand() * (c_max - c_min) + c_min
    hue_factor = (2 * np.random.rand() - 1) * h
    gamma_factor = [1.2, 1, 0.9][np.random.randint(3)]
    data = color_jitter_(data, brightness_factor, saturation_factor, contrast_factor,
                         hue_factor, gamma_factor)
    return data


def transform(data):
    data = gaussian_blur(data)
    data = color_jitter(data)
    return data


def coerce_to_path_and_check_exist(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError('{} does not exist'.format(path.absolute()))
    return path


def coerce_to_path_and_create_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
