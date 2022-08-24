# Umap-based-visualization-for-feature-space-of-self-supervised-learning

## Introduction
You can use this program to observe the feature space of trained checkpoint by self-supervised lerannig.


## Requirements
```
GPU
numpy = 1.22.3
umap-learn = 0.5.3
```
## Usage

You can use the following command to install the umap tool
- Install python dependencies.
```
pip install numpy==1.22
pip install umap-learn
```
- Preparation work

(1) You should place the checkpoint of self-supervised learning in **checkpoint** fold.

(2) You should place a dataset in **dataset** fold, such as KME ([Baidu Netdisk](https://pan.baidu.com/s/1gLRDYK2lmgoLlZuzLcNIfw?pwd=wfzk ) with code **wfzk**):

dataset

    |___KME
        |___ADI
        |___BACK
        |___DEB
        |___LYM
        |___MUC
        |___MUS
        |___NORM
        |___STR
        |___TUM
        


