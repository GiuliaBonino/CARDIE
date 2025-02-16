# CARDIE: Clustering Algorithm for Image Enhancement

This project presents **CARDIE**, a novel clustering algorithm for automatic image clustering, designed specifically for image enhancement tasks. CARDIE labels images based on their color and luminosity content and clusters them in an unsupervised manner. The algorithm is intended to address the challenges of defining meaningful clusters for image enhancement. Additionally, we propose a new method to quantify luminance distribution modifications induced by image enhancement algorithms.

## Installation

In order to try out CARDIE for yourself, you can clone this repository and install the requirements
```
   git clone https://github.com/GiuliaBonino/CARDIE
   pip install -r requirements.txt
```
## Usage
In order to run CARDIE, the first step is to create the table with all the relevant features, to do so, you should add your data's path to the [create_tabular_dataset.py](https://github.com/GiuliaBonino/CARDIE/blob/master/create_tabular_dataset.py) file, add the output file's path and run:
```
  python create_tabular_dataset.py
```
Then, to cluster these features into classes, change the path of the file containing the extracted features and run:
```
  python dataset_clutering,py
```
Moreover, you can also find:
- [quantify_tone_mapping.py](https://github.com/GiuliaBonino/CARDIE/blob/master/quantify_tone_mapping.py) : quantifies the effect of an Image Enhancement operator on the luminance distribution of your dataset 
- [traditional_tone_mapping.py](https://github.com/GiuliaBonino/CARDIE/blob/master/traditional_tone_mapping.py) : applies tone mapping using the Reinhard Devlin operator
- [dataset_clustering_vgg.py](https://github.com/GiuliaBonino/CARDIE/blob/master/dataset_clustering_vgg.py): clusters the images in your dataset using the features extracted with VGG (useful for comparisons to the ones obtained with CARDIE)

In the notebooks folder, you can find various tests and plots that are present also in the paper.
   
