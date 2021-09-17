# Generic Semantic Segmentation of Historical Maps - Github repository

This is the Github repository of the following Conference paper: **R. Petitpierre, F. Kaplan, and I. di Lenardo, *Generic Semantic Segmentation of Historical Maps*, Proceedings of the Workshop on Computational Humanities Research (CHR 2021), November 17-19, 2021, Amsterdam (NL)**

```
@inproceedings{petitpierre_generic_2021,
  author    = {Petitpierre, R{\'{e}}mi and Kaplan, Fr{\'{e}}d{\'{e}}ric and di Lenardo, Isabella},
  title     = {Generic Semantic Segmentation of Historical Maps},
  booktitle = {Proceedings of the Workshop on Computational Humanities Research (CHR 2021)},
  year      = {2021},
  eventdate = {2021-11-17/2021-11-19},
  publisher = {CEUR},
  address   = {Amsterdam, NL}
}
```

# Abstract

Research in automatic map processing is largely focused on homogeneous corpora or even individual maps, leading to inflexible models. Based on two new corpora, the first one centered on maps of Paris and the second one gathering maps of cities from all over the world, we present a method for computing the figurative diversity of cartographic collections. In a second step, we discuss the actual opportunities for CNN-based semantic segmentation of historical city maps. Through several experiments, we analyze the impact of figurative and cultural diversity on the segmentation performance. Finally, we highlight the potential for large-scale and generic algorithms. Training data and code of the described algorithms are made open-source and published with this article.

Keywords: *historical map processing, neural networks, semantic segmentation, figuration, topology*

# Description of the code structure
 - **OperationalizeFiguration.ipynb**: This Jupyter notebook allows to compute the κ-coefficient, which describes the figurative diversity of a corpus of maps
 - **tsneProjection.ipynb**: This Jupyter notebook allows to reproduce the t-SNE projection of the descriptive features of the figuration
 - **textureRemoval.ipynb**: This notebook allows to pre-process the image patches, for the experiment on the importance of graphical cues for learning
 - **utils/descriptors.py**: This script contains the functions needed to calculate the figurative features and therefore run the notebooks
 - **export**: This folder contains the pre-computed features for the two datasets (Paris and World) and the 3 comparison datasets (Napoleonic cadastre, ICDAR2021, and USGS)

### Utils
 - This folder contains some custom Python functions used in the above notebooks.

## Additional resources
As mentioned in the article, we used dhSegment-torch for all CNN-based semantic segmentation experiments, training and inference. We invite you to refer to the following Github repository: [dhlab-epfl/dhSegment-torch](https://github.com/dhlab-epfl/dhSegment-torch).

# Dataset

The World and the Paris datasets (*Historical City Maps Semantic Segmentation Dataset*) are published in open-source and can be dowloaded [here](https://zenodo.org/record/5497934):

```
@misc{petitpierre_historical_2021,
  author    = {Petitpierre, R{\'{e}}mi},
  title     = {Historical City Maps Semantic Segmentation Dataset},
  year      = {2021},
  howpublished = {\url{https://zenodo.org/record/5513639}},
  doi       = {10.5281/zenodo.5513639}
}
```

# License & Liability

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/deed.en). We do not assume any liability for the use of this code.
