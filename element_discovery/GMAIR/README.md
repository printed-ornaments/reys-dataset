# GMAIR

This code has been adapted from [the official repository]((https://github.com/EmoFuncs/GMAIR-pytorch)) of GMAIR.

## Preparation
This project uses Python 3.8, Pytorch 1.8.1 and higher.

```
pip install -r requirements.txt
```

Build bbox:

```
cd GMAIR-pytorch/gmair/utils/bbox
python setup.py build
cp build/lib/bbox.so .
```


## Training
To train the model,

- update the dataset path in `config.py`, and
- run the following command:

```
python train.py
```

## Test
To test the model,
- set the path of checkpoint file in the configuration file `config.py`, and
- run the following command:

```
python test.py
```
