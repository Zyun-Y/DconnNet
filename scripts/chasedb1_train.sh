#! /bin/sh
cd ..
python train.py \
--dataset 'chase' \
--data_root '/home/ziyun/Desktop/project/DRIVE/CHASEDB1' \
--resize 960 960 \
--num-class 1 \
--batch-size 4 \
--epochs 130 \
--lr 0.004 \
--lr-update 'step' \
--lr-step 40 \
--gamma 0.3 \
--folds 5


