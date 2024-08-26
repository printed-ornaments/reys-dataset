## Element Discovery

### Dataset :scroll:

The composite ornaments dataset contains the following files:
- a set of 100 composite ornaments, composed of smalles typographic elements called vignettes. There are 72 such disticnt elements that compose the 100 composite ornaments.
- the minimal dictionary of 72 vignettes used to form the composite ornaments, 
- a set of 100 text files giving each bounding boxes and vignette names per composite ornament in the original images,
- a set of synthetically generated composite ornaments,
- the set of images of composite ornaments rescaled to 128x128,

and as the annotation was performed after zooming the images in 512x512 pixels:
- the set of images of composite ornaments rescaled to 512x512,
- the set of 100 txt files giving each bounding boxes and vignette names per composite ornament in the rescaled images,
organized as follows:

```
./
├── images/    	        # (100 jpeg images of composite ornaments)
├── vignette_dict/      # (72 jpg images of vignettes used to composed the set of the ornaments)
├── annotation/         # (100 txt files listing each bounding boxes surrounding the vignettes identified in the dictionary, used to design the current composite ornament)
├── README.md           # (current file)
├── synt_images/        # (50436 syntheticly generated composite ornaments)
├── crop_128x128/       # (100 jpeg images of composite ornaments, rescaled to 128x128)
├── images_512x512/     # (100 jpeg images of composite ornaments, rescaled to 512x512)
└── annotation_512x512/ # (100 annotation txt files where bounding boxes are given rescaled to 512x512) 
```

Concerning the naming of vignette extracted from a catalog, here are two examples:
- `1768E67_F1AUG1L010_01.jpg` is a sample coming from the Enschede catalog, 
- `1768R133_F1CIC1L009_01.jpg` is a  sample coming from the Rosart catalog. 

The year of the catalog is first reported, secondly the first letter of the catalog name (E for Enschede and R for Rosart, respectively), then the page number (67 and 133 in the examples above) in the ancient book. A "F" is used for mentioning the font, followed by a number giving the number of the font in the page. The font is then indicated by the first three characters of the font name followed by a "0" for a title font, "1" for a simple font, "2" for a double or "3" for a triple one (AUG1 and CIC1 in the examples above). The letter L coming after is the initial of 'Line', followed by three numbers which are the number of the line in the page (L010 or L009). After the second underscore, two numbers are here to specify the position of the vignette in the line (01 for both examples above).
The printing vignettes were named as "V_" followed by an order number encoded on two digits and ".jpg": from `V_01.jpg` to `V_21.jpg`.
Each annotation file has the same radical as the corresponding image file name. It structured in lines with a number of lines equal to the number of vignettes in the current composite ornament. Each line gives first the bounding box then the vignette name, forming all 5 columns structured as follows:
- row then column of the top left corner, followed by height and width of the current bounding box (the first 4 columns),
- followed by the name of the vignette in the dictionary in the last column.

See for instance the content of `BML_301565~~73383.txt`:

```
5 6 32 50 1768E67_F1PAR1L003_01
6 56 31 48 1768E67_F1PAR1L003_02
38 48 22 15 1768E69_F1AUG1L020_01
```
Dataset can be downloaded from the link: [element_discovery.zip](https://drive.google.com/file/d/1A5IpzCsk4MGjZhrXSE8cwsR2f40uoRd8/view?usp=sharing)

and extracted to corresponding directory with following commands:

```
unzip element_discovery.zip && rm element_discovery.zip
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

To evaluate element discovery results, refer to the demo [```element_discovery_eval.ipynb```](https://github.com/mathieuaubry/roii/blob/main/element_discovery/element_discovery_eval.ipynb). To visualize your outputs, refer to [```element_discovery_visualize.ipynb```](https://github.com/mathieuaubry/roii/blob/main/element_discovery/element_discovery_vosualize.ipynb)



### Baselines :seedling:

Here, we present the baselines and a guideline to reproduce the results under corresponding directories. 

| | | Real training data | | Synt. training data | |
| ------- | :------- | :------- | :------- | :------- | :------- |
| Method | Bkg. | AP(%) &#8593; | mAP(%) &#8593; | AP(%) &#8593; | mAP(%) &#8593; |
| SPAIR ([Crawford et al.](https://github.com/e2crawfo/auto_yolo)) | &#9746; | 0 | 0 | 0 | 0 |
| GMAIR ([Zhu et al.](https://github.com/EmoFuncs/GMAIR-pytorch)) | &#9746; | 0 | 0 | 0 | 0 |
| SPACE ([Lin et al.](https://github.com/zhixuan-lin/SPACE)) | &#9745; | 0 | 0 | 8.1 | 6.1 |
| AST-argmax ([Sauvalle et al.](https://github.com/BrunoSauvalle/AST)) | &#9745; | 13.6 | 13.2 | 38.4 | 27.6 |
| DTI Sprites ([Monnier et al.](https://github.com/monniert/dti-sprites/)) | &#9745; | 0 | 0 | 0 | 0 |

### License

[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

