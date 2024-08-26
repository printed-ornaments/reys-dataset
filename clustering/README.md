## Clustering

### Dataset :scroll:

The dataset contains the following files:
- a set of 339 images of block ornaments, fleurons printed in the 18th century with engravings in books edited by Marc-Michel Rey, with occurrences groupable together into clusters, one per block ornament,
- metadata about the original books and the cluster labels,
- the present readme file indicating the origin and conditions of use of the data,
organized as follows:

```
./
├── images/                 # (339 jpeg images of block ornaments)
├── metadata.tsv            # (tsv file listing the block ornaments)
├── metadata_filtered.tsv   # (tsv file listing the block ornaments with classes that have more than 3 samples in each class)
├── metadata_balanced.tsv   # (tsv file listing the block ornaments with balanced classes)
└── README.md               # (current file)
```

The annotation file (`metadata.tsv`) has 9 columns filled in line by line for each block ornament. The first and last columns give the file and class names of each block ornament. The other columns include its Heurist index and data linking the image to the original book. The images were renamed according to the following format: `LibraryAcronym_BookIdentifier~~HeuristIndex.jpeg` where HeuristIndex is an index assigned in the framework of the ROIi project. See examples of an image file name in an extract from the annotation file below:

```
image_filename           ROIi_Heurist_ID Edition   genre          nature               notes_ornement_occurr bbthq_calc_v2  xmpl_cote_calc    motif_id    notes_ornement_motif	
BML_104316_2~~72799.jpeg 72799           1774Grav  fleuron bloc   autorisation_Lyon_BM Lyon BM               104316         /2                fl_bloc_0093		
```

<img src="https://imagine.enpc.fr/~sonat.baltaci/roii-images-github/balance.png" width=350pt>

| Dataset | Base | Imbalanced | Balanced |
| :------ | :--- | :--------- | :------- |
| # ornaments | 339 | 167 | 70 |
| # classes | 163 | 36 | 14 |

While we will release the full set and associated annotations, we focused our evaluation on two subsets:
- an imbalanced subset, with all the 167 images from the 36 classes that have at least 3 instances: `metadata_filtered.tsv`,
- a balanced subset, with 70 images, built by randomly sampling 5 images from the 14 classes that have at least 5 instances: `metadata_balanced.tsv` .

Dataset can be downloaded from the link: [clustering.zip](https://drive.google.com/file/d/1QJQOZPuBmndCTmQViNn2xPwBUPxMe4OQ/view?usp=sharing)

and extracted to corresponding directory with following commands:

```
unzip clustering.zip && rm clustering.zip
```

### Evaluation :chart_with_upwards_trend:

#### Environment

To create an evaluation environment, run the following commands consequtively:
```
conda env create -f environment.yml
conda activate roii
```

To install `conda`, refer to [Anaconda Installation Guide](https://docs.anaconda.com/free/anaconda/install/index.html).

#### Evaluation scripts

To evaluate clustering results, refer to the demo [```clustering_demo.ipynb```](https://github.com/mathieuaubry/roii/blob/main/clustering/clustering_demo.ipynb).

### Baselines :seedling:

Here, we present the baselines and a guideline to reproduce the results. The official repositories used are linked within the corresponding method in tables.

#### Clustering over features

| Dataset | Imbalanced | | Balanced | |
| ------- | :------- | :------- | :------- | :------- |
| Method  | Acc. (%) &#8593; | NMI (%) &#8593; | Acc. (%) &#8593; | NMI (%) &#8593; |
| ICC ([Xu et al.](https://github.com/xu-ji/IIC)) | 19.2 | 40.6 | 25.0 | 35.0 |
| SCAN ([Gansbeke et al.](https://github.com/wvangansbeke/Unsupervised-Classification)) | 46.1 | 68.7 | 47.1 | 64.6 |
| DivClust ([Metaxas et al.](https://github.com/ManiadisG/DivClust)) | 54.1±0.5 | 80.5±0.3 | 67.0±2.1 | 78.6±0.7 |
| K-Means w/ DINO ([Caron et al.](https://github.com/facebookresearch/dino)) | 73.9±0.9 | 89.9±0.4 | 70.9±3.3 | 86.8±1.6 |
| K-Means w/ CLIP ([Radford et al.](https://github.com/openai/CLIP)) | 74.6±3.0 | 90.1±1.1 | 78.3±2.9 | 89.8±1.2 |

To reproduce each baseline for imbalanced dataset, run corresponding command(s) within the project directory:
```
# IIC
cp ./baseline_scripts/iic_fleurons.py path_to_IIC/code/utils/cluster/fleurons.py
cd path_to_IIC/ # change directory to project directory
CUDA_VISIBLE_DEVICES=0 python -m code.scripts.cluster.cluster_sobel_twohead --model_ind 0 --arch ClusterNet5gTwoHead --mode IID --dataset Fleurons --dataset_root /home/sonat/dti-sprites/datasets/fleurons --gt_k 36 --output_k_A 42 --output_k_B 36 --lr 0.0001 --num_epochs 3200 --batch_sz 32  --num_dataloaders 4 --num_sub_heads 1 --head_A_first --head_B_epochs 2 --crop_orig --input_sz 96

# SCAN
cp ./baseline_scripts/simclr_fleurons.yml path_to_SCAN/configs/pretext/simclr_fleurons.yml
cp ./baseline_scripts/scan_fleurons.yml path_to_SCAN/configs/scan/scan_fleurons.yml
cp ./baseline_scripts/scan_data_fleurons.py path_to_SCAN/data/fleurons.py
cd path_to_SCAN/ # change directory to project directory
CUDA_VISIBLE_DEVICES=0 python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_fleurons.yml 
CUDA_VISIBLE_DEVICES=0 python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_fleurons.yml

# DivClust
cp ./baseline_scripts/divclust_data_fleurons.py path_to_DivClust/data/dataset_implementations/fleurons.py
cp ./baseline_scripts/cc_fleurons.yml path_to_DivClust/configs/cc_fleurons.yml
cd path_to_DivClust/ # change to project directory
python main.py --gpu 0 --preset cc_fleurons --clusterings 20 --NMI_target .8 --batch_size 32 --NMI_interval 20

# K-Means
# DINO
python baselines/run_kmeans.py --feat dino --ds path_to_dataset/ --n_clusters 36
# CLIP
python baselines/run_kmeans.py --feat clip --ds path_to_dataset/ --n_clusters 36
```

#### Clustering over pixels

| Dataset | Imbalanced | | Balanced | |
| ------- | :------- | :------- | :------- | :------- |
| Method  | Acc. (%) &#8593; | NMI (%) &#8593; | Acc. (%) &#8593; | NMI (%) &#8593; |
| K-Means | 65.6±1.8 | 84.1±1.1 | 74.3±1.9 | 83.9±0.7 |
| DTI K-Means ([Monnier et al.](https://github.com/monniert/dti-clustering)) | 75.7±0.8 | 90.7±0.6 | 87.4±1.9 | 92.1±1.2 |

To reproduce each baseline for imbalanced dataset, run corresponding command(s) within the project directory:
```
# K-Means
python baselines/run_kmeans.py --feat image --ds path_to_dataset/ --n_clusters 36

# DTI-Clustering
cp ./baseline_scripts/dti_fleurons.yml path_to_DTI/configs/fleurons.yml
cp ./baseline_scripts/dti_data_init.py path_to_DTI/src/dataset/__init__.py
cp ./baseline_scripts/dti_fleurons.py path_to_DTI/src/dataset/fleurons.py
cd path_to_DTI/
cuda=0 config=fleurons.yml tag=roii_clustering ./pipeline.sh
```


### License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

