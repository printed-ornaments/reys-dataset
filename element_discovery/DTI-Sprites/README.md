# DTI-Sprites

To preproduce DTI-Sprites baseline,

- run following commands within the project directory to setup the environment:

```
cp ./baseline_scripts/fleuron_compounds_synt_tune.yml path_to_DTI/configs/fleuron_compounds_synt_tune.yml
cp ./baseline_scripts/fleuron_compounds_synt.yml path_to_DTI/configs/fleuron_compounds_synt.yml
cp .baseline_scripts/__init__.py path_to_DTI/src/dataset/__init__.py
cp ./baseline_scripts/multi_object.py path_to_DTI/src/dataset/multi_object.py
```

- run the following command to train the model on synthetic dataset:

```
cuda=0 config=fleuron_compounds_synt.yml tag=roii_synt ./pipeline.sh
```

- update the path to the pretrained model in the config file `fleuron_compounds_synt_tune.yml` and run the following command to tune the model on real data:

```
cuda=0 config=fleuron_compounds_synt_tune.yml tag=roii_synt_tune ./pipeline.sh
```