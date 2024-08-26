# AST-argmax

The code is adapted from [the original repository](https://github.com/BrunoSauvalle/AST) of AST.

The AST-argmax model has two parts. A background model as a convolutional autoencoder, followed by a foreground model.

Before training run following command for logging:

```
cd AST
mkdir output && mkdir results && mkdir results/background_output
```

## Create an environment

First you need to create a conda environment and install all the necessary requirements on it.
```
conda create -n AST
conda activate AST
pip install -r requirements.txt
```


## Training

The steps to the train AST-argmax are as follows:

### Step 1: Train the background model

To train the background model,

- go to the background directory and create a new entry in the background configuration file ```config.py``` with the path to the images dataset,
 
```
cd background/
```
   
- and start training with the following command:

```
python train.py
```

### Step 2: Generate the background dataset
To generate the dataset,

- update the background configuration file with the path to the final background checkpoint which can be found in  the background model output directory, and
- start to generate the background dataset with the following command:

```
python generate_background_dataset.py
```

This command will create the following datasets in the background model output directory:

- background image datasets (RGB)
- background image with error prediction dataset (RGBA)
- copy of the input image dataset as a frame sequence (with the same ordering as the background images)
- copy of ground-truth segmentation masks if available in the input dataset
- background/foreground segmentation masks predicted by the background model

### Step 3: Train the foreground model
To train the foreground model,

- create a new entry in the foreground configuration file ```MF_config.py```, and paths to the required datasets in the background model output directory,
- start training the foreground model with the following command:

```
python MF_train.py
```

During training, some image samples are generated and to the path ```training_images_output_directory``` set in the configuration file ```MF_config.py``` which have to be updated. 
    

### Step 4: Run the trained foreground model to get segmentation masks

To get the segmentation masks,

- update the foreground configuration file ```object_model_checkpoint_path``` with the path to the trained model, and
- generate samples with the following command:

```
python MF_generate_image_samples.py
```

## Evaluation

In order to load the pretrained model and reproduce the qualitative results:

- put the path of the dataset in the ```MF_config.py``` file along with the path of the pre-trained model,
- make sure you have the same name for the experiment as you have in the `background/config.py` file, and
- run `MF_generate_image_samples.py` to get qualitative samples.
