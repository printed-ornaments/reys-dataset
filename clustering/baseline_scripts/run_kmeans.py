import argparse
import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from skimage import io
import clip

from src.utils.metrics import Scores


def feat_extractor(name, data):
    if name == "image":  # No feature extraction, optimize over image pixels
        images = np.array(data)
        n_samples, height, width, n_channels = images.shape
        images = images.reshape((n_samples, height * width * n_channels))
        scaler = StandardScaler()
        feats = scaler.fit_transform(images)
    elif name == "dino":  # Extract DINO ViTS8 features
        dino = torch.hub.load("facebookresearch/dino:main", "dino_vits8").eval()
        data = torch.Tensor(np.array(data))
        B, H, W, C = data.shape
        feats = dino(data.reshape(B, C, H, W)).detach().numpy()
    elif name == "clip":  # Extract CLIP ViTB32 features
        clip_model, preprocess = clip.load("ViT-B/32", device="cpu")  # run on cpu
        image = [preprocess(Image.fromarray(np.uint8(i * 255))) for i in data]
        with torch.no_grad():
            feats = clip_model.encode_image(torch.stack(image, dim=0)).detach().numpy()
    else:
        raise NotImplementedError("Feature extractor isn't implemented.")
    return feats


def prep_data(root):
    data_dir = root

    # Load annotations (for imbalanced, filtered subset)
    metadata_path = os.path.join(
        data_dir, "metadata_filtered.tsv"
    )
    annotations = pd.read_csv(metadata_path, delimiter="\t")
    labels, _ = pd.factorize(annotations["motif_id"])
    image_paths = [
        os.path.join(data_dir, "images", filename)
        for filename in annotations["image_filename"]
    ]
    images = []

    for img_path in image_paths:
        img = io.imread(img_path)
        img = resize(img, (256, 256, 3))
        images.append(img)

    return images, labels


def kmeans(n_clusters, images, n_iter, batch_size):
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        max_iter=n_iter,
        random_state=0,
        init="k-means++",
        batch_size=batch_size,
    )
    return kmeans.fit(images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to run k-Means over images.")

    parser.add_argument(
        "--n_clusters", "-n", type=int, default=36, help="Number of clusters"
    )

    parser.add_argument(
        "--dataset_path",
        "--ds",
        nargs="?",
        type=str,
        required=True,
        help="Dataset path",
    )

    parser.add_argument(
        "--feat",
        nargs="?",
        type=str,
        required=True,
        help="Feature extractor",
    )

    parser.add_argument(
        "--n_iter", type=int, default=300, help="Max number of iterations"
    )

    parser.add_argument("--batch_size", type=int, default=167, help="Batch size")

    args = parser.parse_args()

    # Prepare data
    data = prep_data(args.dataset_path)

    # Extract features
    feats = feat_extractor(args.backbone, data[0])

    # Run k-Means
    model = kmeans(args.n_clusters, feats, args.n_iter, args.batch_size)
    labels_pred = model.predict(feats)

    # Compute metrics
    scores = Scores(args.n_clusters, args.n_clusters)
    scores.update(data[1], labels_pred)
    scores = scores.compute()
    print(
        "Final_scores: "
        + ", ".join(["{}={:.4f}".format(k, v) for k, v in scores.items()])
    )
