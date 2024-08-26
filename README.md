# Historical Printed Ornaments: Dataset and Tasks

Official repository of the paper _"Historical Printed Ornaments: Dataset and Tasks"_. We introduce **_Rey's Ornaments dataset_** which focuses on an XVIIIth century bookseller, Marc Michel Rey, providing a consistent set of ornaments with a wide diversity and representative challenges. We additinally highlight three complex tasks that are of critical interest to book historians: (a) clustering, (b) element discovery, and (c) unsupervised change localization.

<p align='center'>
<img src="https://imagine.enpc.fr/~sonat.baltaci/roii-images-github/teaser.png" height="300pt">
</p>

## Datasets :scroll: and Evaluation :chart_with_upwards_trend:

We offer three task-specific subsets of **_Rey's Ornaments dataset_**:

```
./
├── clustering
├── element_discovery
└── change_localization
```
Detailed explanations of datasets, tasks, baselines and evaluation demos are presented in following ReadMe files:
- clustering [ReadMe](https://github.com/mathieuaubry/roii/blob/main/clustering/README.md)
- element discovery [ReadMe](https://github.com/mathieuaubry/roii/blob/main/element_discovery/README.md)
- change localization [ReadMe](https://github.com/mathieuaubry/roii/blob/main/change_localization/README.md)

## Citation :bookmark:

```
@inproceedings{chaki2024historical,
  title={Historical Printed Ornaments: Dataset and Tasks},
  author={Chaki, Sayan Kumar and Baltaci, Zeynep Sonat and Vincent, Elliot and Emonet, Remi and Vial-Bonacci, Fabiene and Bahier-Porte, Christelle and Aubry, Mathieu and Fournel, Thierry},
  booktitle={ICDAR},
  year={2024}
}
```

## License
[![CC BY 4.0][cc-by-shield]][cc-by]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg

## Acknowledgements
The **_Rey's Ornaments dataset_** was produced as part of the **Rey’s Ornaments Image investigation** project (ROIi, 2020-2025, see <https://ro2i.hypotheses.org>), a project in digital humanities funded by the French National Research Agency (ANR). The dataset which is a part of the ROIi database, is made publicly  available for research purposes in computer vision by the members of the ROIi project, IHRIM ([UMR 5317](https://ihrim.ens-lyon.fr/)) and Hubert Curien lab ([UMR 5516](https://laboratoirehubertcurien.univ-st-etienne.fr/)). 

The images were cropped from digitized pages of ancient books, stored either in the Bibliothèque Nationale de France (BnF) or in the Bibliothèque Municipale de Lyon (BML, a partner of the ROIi project). The provenance for each image file is indicated by the first part of the file name (« BnF » and « BML », respectively). Their conditions of use are indicated by the license files accessible on the websites of [BnF Gallica](https://gallica.bnf.fr/edit/und/conditions-dutilisation-des-contenus-de-gallica) and [BML Numélyo](https://numelyo.bm-lyon.fr/conditions_utilisation).

S. Baltaci, E. Vincent and M. Aubry were supported by ERC project DISCOVER funded by the European Union’s Horizon Europe Research and Innovation program under grant agreement No. 101076028 and ANR VHS project ANR-21-CE38-0008.
   
