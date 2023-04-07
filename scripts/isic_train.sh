#! /bin/sh
cd ..
python train.py \
--dataset 'isic' \
--data_root '/home/ziyun/Desktop/project/ISIC/codes/2017/data/ISIC2018_npy_all_224_320' \
--resize 224 320 \
--num-class 1 \
--batch-size 16 \
--epochs 150 \
--lr 1.5e-4 \
--lr-update 'CosineAnnealingWarmRestarts' \
--folds 5

