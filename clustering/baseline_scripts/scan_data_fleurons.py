import torch
import os
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class FleuronsDataset(Dataset):
    def __init__(self, dataset_path=None, transform=None) -> None:
        self.n_classes = 36
        if dataset_path is None:
            self.data_path = "../roii/clustering/"
        else:
            self.data_path = dataset_path

        metadata_path = os.path.join(
            self.data_path, "metadata_filtered.tsv"
        )
        img_path = os.path.join(self.data_path, "images")
        annotations = pd.read_csv(metadata_path, delimiter="\t")
        assert os.path.exists(os.path.join(img_path))
        self.files = [
            os.path.join(img_path, filename)
            for filename in annotations["image_filename"]
        ]
        labels, _ = pd.factorize(annotations["motif_id"])
        self.classes = list(set(annotations["motif_id"]))
        self.labels = torch.Tensor(labels)

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def get_image(self, index):
        img = np.array(Image.open(self.files[index]).convert("RGB"))
        return img

    def __getitem__(self, idx):
        img = np.array(Image.open(self.files[idx]).convert("RGB"))
        target = self.labels[idx]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        class_name = self.classes[int(target.item())]

        if self.transform is not None:
            img = self.transform(img)

        out = {
            "image": img,
            "target": target,
            "meta": {"im_size": img_size, "index": idx, "class_name": class_name},
        }
        return out
