# Directional Connectivity-based Segmentation of Medical Images
**Pytorch implementation for CVPR2023 paper "Directional Connectivity-based Segmentation of Medical Images" [[paper](https://arxiv.org/pdf/2304.00145.pdf)].**

## Requirements
Pytorch 1.7.0+cu110

## Implementation
The main stucture (important .py files) of this repository is as following:
```
general
  - train.py: main file. Define your parameters, selection of GPU etc.
  - solver.py: the training details and testing details.
  - connect_loss.py: loss function for DconnNet
  
  data_loader: your data loader files and the SDL weights for your dataset if needed.
  model: DconnNet model files
```
