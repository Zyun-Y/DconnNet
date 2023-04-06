# Directional Connectivity-based Segmentation of Medical Images
**Pytorch implementation for CVPR2023 paper "Directional Connectivity-based Segmentation of Medical Images" [[paper](https://arxiv.org/pdf/2304.00145.pdf)].**

## Requirements
Pytorch 1.7.0+cu110

## Code Stucture
The main stucture (important .py files) and important functions of this repository is as following:
```
general
  - train.py: main file. Define your parameters, selection of GPU etc.
  - solver.py: the training details and testing details.
  - connect_loss.py: loss function for DconnNet
    * connectivity_matrix: converting segmentation masks to connectivity masks
    * Bilateral_voting: bilateral voting and convert connectivity-based output into segmentation map.
    
  data_loader: your data loader files and the SDL weights for your dataset if needed.
  model: DconnNet model files
  scripts: bash files for training different datasets
```
## Implementation
### Training on datasets in the paper.

### Training on your own dataset.
 1. Make your own dataloader.
 2. Replace your dataloader in main() function of ```train.py```. If need k-fold validation, use exp_id to specify your sub-folds.
 3. Specify your network setting in ```train.py```
 4. Run the following command
  ```python train.py```
  
## Pretrained model
