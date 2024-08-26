# SPACE

The code is adapted from [the official repository](https://github.com/zhixuan-lin/SPACE) of SPACE.

## General

Project directories:

* `src`: source code
* `data`: where you should put the datasets
* `output`: anything the program outputs will be saved here. These include
  * `output/checkpoints`: training checkpoints. Also, model weights with the best performance will be saved here
  * `output/logs`: tensorboard event files
  * `output/eval`: quantitative evaluation results
  * `output/demo`: demo images
* `scripts`: some useful scripts for downloading things and showing demos
* `pretrained`: where to put downloaded pretrained models

This project uses [YACS](https://github.com/rbgirshick/yacs) for managing experiment configurations. Configurations are specified with YAML files. These files are in `src/configs`. We provide five YAML files that correspond to the figures in the paper:

* `vignettes.yaml`: for the Rey's Ornaments dataset.

## Dependencies

This project uses Python 3.7 and PyTorch 1.3.0. and higher

Create a conda environment with Python 3.7 and activate it. Other versions of Python should also be fine:

```
conda create -n space python=3.7
conda activate space
```

Install PyTorch 1.3.0:

```
pip install torch==1.3.0+cu100 torchvision==0.4.1+cu100 -f https://download.pytorch.org/whl/torch_stable.html
```

Note that this requires CUDA 10.0. If you need CUDA 9.2 then change `cu100` to `cu92`. Depending on your cuda version, you may want to install previous versions of PyTorch.  See [here](https://pytorch.org/get-started/previous-versions/).

Other requirements are in `requirements.txt` and can be installed with

```
pip install -r requirements.txt
```
## Datasets

Put the downloaded datasets under the `data` directory.

## Training and Evaluation

To train the model,
-  first, `cd src`.  Make sure you are in the `src` directory for all commands in this section. All paths referred to are also relative to `src`.

The general command to run the program is (assuming you are in the `src` directory) is as follows:

```
python main.py --task [TASK] --config [PATH TO CONFIG FILE] [OTHER OPTIONS TO OVERWRITE DEFAULT YACS CONFIG...]
```

To train your model on our dataset, run the following command:

  ```
  python main.py --task train --config configs/vignettes.yaml resume True device 'cuda:0'
  ```


