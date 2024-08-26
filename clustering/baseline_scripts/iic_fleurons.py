import torch
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class FleuronsDataset(Dataset):
    def __init__(self, **kwargs) -> None:
        self.transform = kwargs.get("transform")
        self.target_transform = kwargs.get("target_transform")
        self.n_classes = 36
        self.data_path = kwargs.get(
            "config.dataset_root", "../roii/clustering/"
        )

        self.img_size = kwargs.get("img_size", 128)
        self.n_channels = 3
        if type(self.img_size) is int:
            self.img_size = (self.img_size, self.img_size)
        elif len(self.img_size) == 2:
            self.img_size = self.img_size
        else:
            raise ValueError

        try:
            metadata_path = os.path.join(
                self.data_path,
                "metadata_filtered.tsv",
            )
            img_path = os.path.join(self.data_path, "images")
            annotations = pd.read_csv(metadata_path, delimiter="\t")
            self.files = [
                os.path.join(img_path, filename)
                for filename in annotations["image_filename"]
            ]
            self.labels, _ = pd.factorize(annotations["motif_id"])
        except FileNotFoundError:
            self.files = []

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = self.transform(Image.open(self.files[idx]).convert("RGB"))
        target = (
            self.target_transform(self.labels[idx])
            if self.target_transform
            else self.labels[idx]
        )

        return img, target
