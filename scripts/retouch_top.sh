#! /bin/sh
cd ..
python train.py \
--dataset 'retouch-Topcon' \
--data_root '/home/ziyun/Desktop/project/retouch/dataset' \
--resize 256 256 \
--num-class 4 \
--batch-size 8 \
--epochs 50 \
--lr 0.0008 \
--lr-update 'step' \
--use_SDL  \
--folds 3

