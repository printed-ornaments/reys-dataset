## Unsupervised Change Localization

### Dataset :scroll:

This dataset contains 30 folders corresponding to 30 vignettes from three catalogs: 22 in the Enschede, 2 in the Rosart, and 6 in the Fournier. Each folder is divided into 3 subfolders named `normal`, `changed`, `unchanged` containing the following files:
- `normal`: 4 images said normal i.e. without defect altering the perception of the underlying glyph, all extracted from a catalog,
- `changed`: 1 image with detectable changes altering the perception of the underlying glyph, extracted from the same catalog, 1 text file listing the bounding boxes surrounding the changes with respect to the normal images, usually a lack (negative change) or excess (positive change) of ink, and 1 tif color image where the signed changes are segmented,
- `unchanged`: 1 image with with no detectable change.

The data is organized as follows:
```
./
├── _vignette_name_/         #(image and annotation file of _vignette_name_)
	├── normal/          #(4 jpg normal images)
	├── changed/         #(1 jpg changed image + corresponding `.tif` and `.txt` annotation files)
	└── unchanged/       #(1 jpg unchanged images) 
```

For each vignette, the two annotation files in the `changed` subfolder have the same radical as the file name of the image with changes. The txt file which has the number of changes in the first line, is structured in lines from line 2, with 5 columns formatted as follows:
- row then column of the top left corner, followed by height and width of the current bounding box surrounding the corresponding change (the first 4 columns),
- followed by the sign of the change in the last column.
The tif file is a RGB file where the pixels of the negative changes are set to 255 in the R plane, the positive ones to 255 in the B plane (the isolated pixels were removed).

Dataset can be downloaded from the link: [change_localization.zip](https://drive.google.com/file/d/1hwWvKlyC2kKxxDuN79vcq7OuOi-Xa55K/view?usp=sharing)

and extracted to corresponding directory with following commands:

```
unzip change_localization.zip && rm change_localization.zip
```

### Evaluation :chart_with_upwards_trend:

#### Environment

To create an evaluation environment, run the following commands consequtively after cloning the repository:
```
cd ./change_localization/
python3 -m venv reyschangeloc
source reyschangeloc/bin/activate
python3 -m pip install -r requirements.txt
```
This implementation uses Pytorch.

#### Evaluation scripts

To train and evaluate a baseline, run:
```
mkdir results
PYTHONPATH=$PYTHONPATH:./src python src/trainer.py -c <method_name>.yaml
```
where <method_name> is `naive`, `congealing` or `vae`.

The code to reproduce STAE baseline will be provided soon.

### Baselines :seedling:

Here, we present the baselines and a guideline to reproduce the results. 

| Method | Naive | | VAE [3] | | STAE [1] | | Cong. [2] | |
| ------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- | :------- |
| Category | C | CU | C | CU | C | CU | C | CU |
| dot1 | 0.0 | 0.0 | 2.5 | 1.8 | 5.2 | 2.5 | 7.5 | 5.0 |
| dot2 | 18.0 | 8.9 | 9.7 | 7.3 | 14.7 |0.7 | 38.5 | 35.7 |
| dot3 | 19.8 | 7.5 | 18.2 | 9.3 | 23.4 | 5.7 | 36.4 | 28.4 |
| dot4 |1.6|0.0|5.0|5.4|4.7|1.2|0.0|0.0|
| dot5 |18.2|20.0|16.7 |14.5 |24.5| 31.3 |21.8 |23.5|
| emblem1 | 27.1 | 21.7 | 22.8 | 16.3 | 2.2 | 2.5 | 32.1 | 26.3 |
| emblem2 | 28.9 | 22.8 |37.4 |26.0| 71.0 |50.0| 62.0| 47.9|
|emblem3 |35.8 |29.0 |29.5 |18.2| 40.0| 0.6| 52.5 |42.0|
|emblem4 |2.3 |1.0| 1.5| 1.1| 1.7| 2.2 |21.2| 20.0|
|emblem5 |21.3| 16.4| 15.1| 11.2| 28.4| 1.3| 26.1| 23.6|
|flower1 |0.7 |0.9| 1.7| 1.5| 0.9| 0.0 |15.0 |13.6|
|flower2 |41.2| 21.1| 36.5| 19.0| 25.0| 0.8 |50.0| 45.9|
|flower3 |4.7 |5.2| 7.7| 7.5| 26.2| 0.0| 21.8| 19.3|
|flower4 |2.8 |2.4| 1.3 |1.3| 24.1 |2.1 |28.1 |27.9|
|flower5 |28.6 |5.4| 18.1| 5.7| 43.4| 27.3| 28.9| 21.1|
|interlacing1| 1.2| 0.8| 1.1| 0.6| 11.9| 0.0| 45.7| 12.6|
|interlacing2| 39.8| 27.1| 30.5 |20.4| 27.6| 15.9 |36.0 |26.1|
|interlacing3| 31.3| 25.0| 40.2| 31.7| 21.9| 0.0| 22.8| 14.6|
|interlacing4| 32.4| 17.6| 39.7| 18.1| 35.5| 16.0| 48.0| 23.5|
|interlacing5| 2.8| 1.3| 3.1| 1.8| 15.6 |0.3 |24.7| 7.3|
|ring1| 18.5 |14.8| 17.7| 10.3| 8.3| 4.3| 37.9| 29.8|
|ring2| 13.5| 11.1| 15.9| 11.8| 27.2| 19.1| 23.3| 17.5|
|ring3| 0.6| 0.6| 1.4| 1.2|1.9| 1.5| 22.9| 13.4|
|ring4| 1.5 |1.5| 6.3 |4.5| 3.9| 3.8 |10.1 |7.5|
|ring5| 0.5| 0.8 |1.4| 1.5 |0.0| 0.0| 5.4| 5.1|
|symbol1| 0.0 |0.0 |3.3| 1.9 |4.9 |0.0 |7.3 |5.4|
|symbol2| 8.4 |3.4 |13.6 |2.9 |7.4 |6.8| 16.7 |11.9|
|symbol3| 2.2 |1.2 |13.0 |10.6| 29.9| 14.6 |42.7 |19.0|
|symbol4| 11.9| 11.3| 8.3| 6.9 |19.0 |5.1 |16.2 |14.9|
|symbol5| 0.8 |0.5 |5.1| 3.3| 22.9 |23.5 |20.3 |16.7|
| __mIOU__ | 13.9 | 9.3 | 14.2 | 9.1 | 19.1 | 8.0 | 27.4 | 20.2 |

### Bibliography

[1] Chaki, S., Baltaci, S., Vincent, E., Emonet, R., Vial-Bonacci, F., Bahier-Porte, C., Aubry, M. & Fournel, T. (2024). Historical Printed Ornaments: Dataset and Tasks. In Submission.  
[2] Cox, M., Sridharan, S., Lucey, S., Cohn, J. (2008). Least squares congealing for unsupervised alignment of images. In: 2008 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2008.  
[3] Kingma, D.P., Welling, M. (2014). Auto-encoding variational bayes. In: 2nd International Conference on Learning Representations, ICLR 2014.

### License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

