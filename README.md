# Directional Connectivity-based Segmentation of Medical Images
**Pytorch implementation for CVPR2023 paper "Directional Connectivity-based Segmentation of Medical Images" [[paper](https://arxiv.org/pdf/2304.00145.pdf)].**

For another simple connectivity-based method, please also check [BiconNet](https://github.com/Zyun-Y/BiconNets)

![image](https://user-images.githubusercontent.com/72995945/230514751-29287ab6-a226-495e-99c7-fcf03254f027.png)

## Requirements
Pytorch 1.7.0+cu110

## Code Stucture
The main stucture and important files or functions of this repository is as following:
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
### Train on datasets in the paper.
For training detail of each dataset, please check the ```general/scripts/```

Please store the each dataset in the following path:

**Retouch**
```
dataset
  /Cirrus ### device, same for Spectrailis and Topcon
    /train
      /TRAIN002 ### volume id
        /mask ### store .png masks here
        /orig ### store .png images here
```

**ISIC2018**

Please download the resized data at [here](https://drive.google.com/drive/folders/1Jz-4GP72ymEX5AOcQj8uKZ9y5TP5uYWo?usp=share_link)
```
/ISIC2018_npy_all_224_320
  /image
  /label

```

**CHASEDB1**
```
/CHASEDB1
  /img
  /gt
```

### Train on your own dataset using this code.
 1. Make your own *dataloader*.
 2. Replace your dataloader in main() function of ```train.py```. If need k-fold validation, use *exp_id* to specify your sub-folds.
 3. Specify your network setting in ```train.py```
 4. Run: 
  ```python train.py```

### Train DconnNet on your own codes. 
**Important: please follow these steps to ensure you get a correct implementation**
 1. Get our model files from ```/model``` 
 2. In the training phase, please use ```connect_loss.py``` as the loss function
    * for single-class, use ```connect_loss.single_class_forward```
    * for general multi-class,use ```connect_loss.multi_class_forward```

 4. In the testing phase,  please **follow our official procedure** in ```test_epoch``` of ```/solver.py``` based on the number of your classes.
    * for single-class, we get the final predictions by ```sigmoid --> threshold --> Bilateral_voting```
    * for general multi-class, we get the final predictions by ```Bilateral_voting --> topK (softmax + topK) ```
    * you might also need to create two variable ```hori_translation``` and ```verti_translation``` in this step for matrix shifting purpose, you can follow the codes or customize your own shifting methods.
  

## Notice of the codes
Please always make sure the dimenstion of your data is correct. For, example, in ```connect_loss.py```, we specified the shape of each each in the comment. When there is issue, please always check the dimension first.

## Pretrained model
The pretrained model can be downloaded at [here](https://drive.google.com/drive/folders/11EcwtsgaSx63ZQVLLj4c7-Nr9Wpok6BT?usp=sharing)

## 

## Citation
If you find this work useful in your research, please consider citing:

*Z. Yang and S. Farsiu, "Directional Connectivity-based Segmentation of Medical Images" in CVPR, 2023.*
