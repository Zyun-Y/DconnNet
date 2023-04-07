import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from data_loader.GetDataset_ISIC2018 import ISIC2018_dataset
from data_loader.GetDataset_Retouch import MyDataset
from data_loader.GetDataset_CHASE import MyDataset_CHASE
from model.DconnNet import DconnNet
import glob
import argparse
from torchvision import datasets, transforms
from solver import Solver
import torch.nn.functional as F
import torch.nn as nn
# from GetDataset import MyDataset
import cv2
from skimage.io import imread, imsave
import os

torch.cuda.set_device(2) ## GPU id

def parse_args():
    parser = argparse.ArgumentParser(description='DconnNet Training With Pytorch')

    # dataset info
    parser.add_argument('--dataset', type=str, default='retouch-Spectrailis',  
                        help='retouch-Spectrailis,retouch-Cirrus,retouch-Topcon, isic, chase')

    parser.add_argument('--data_root', type=str, default='/retouch',  
                        help='dataset directory')
    parser.add_argument('--resize', type=int, default=[256, 256], nargs='+',
                        help='image size: [height, width]')

    # network option & hyper-parameters
    parser.add_argument('--num-class', type=int, default=4, metavar='N',
                        help='number of classes for your data')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=45, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.00085, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-update', type=str, default='step',  
                        help='the lr update strategy: poly, step, warm-up-epoch, CosineAnnealingWarmRestarts')
    parser.add_argument('--lr-step', type=int, default=12,  
                        help='define only when you select step lr optimization: what is the step size for reducing your lr')
    parser.add_argument('--gamma', type=float, default=0.5,  
                        help='define only when you select step lr optimization: what is the annealing rate for reducing your lr (lr = lr*gamma)')

    parser.add_argument('--use_SDL', action='store_true', default=False,
                        help='set as True if use SDL loss; only for Retouch dataset in this code. If you use it with other dataset please define your own path of label distribution in solver.py')
    parser.add_argument('--folds', type=int, default=3,
                        help='define folds number K for K-fold validation')

    # checkpoint and log
    parser.add_argument('--pretrained', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--weights', type=str, default='/home/ziyun/Desktop/project/BiconNet_codes/DconnNet/general/data_loader/retouch_weights/',
                        help='path of SDL weights')
    parser.add_argument('--save', default='save',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--save-per-epochs', type=int, default=15,
                        help='per epochs to save')

                        
    # evaluation only
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='test only, please load the pretrained model')
    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    return args


def main(args):
    
    ## K-fold cross validation ##
    for exp_id in range(args.folds):

        if args.dataset == 'isic':
            trainset = ISIC2018_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='train', 
                                           with_name=False)
            validset = ISIC2018_dataset(dataset_folder=args.data_root, folder=exp_id+1, train_type='test',
                                               with_name=False)
        elif 'retouch' in args.dataset:   
            device_name = args.dataset.split('-')[1]
            path = args.data_root+ '/'+device_name +'/train'
            pat_ls = glob.glob(path+'/*')

            ### for Cirrus
            if device_name == 'Cirrus':
                total_id = [i for i in range(24)]
                test_id = [i for i in range(exp_id*8,(exp_id+1)*8)]

            ### for Spectrailis
            if device_name == 'Spectrailis':
                total_id = [i for i in range(24)]
                test_id = [i for i in range(exp_id*8,(exp_id+1)*8)]

            ### for Topcon
            if device_name == 'Topcon':
                total_id = [i for i in range(22)]
                if exp_id<2:
                    test_id = [i for i in range(exp_id*7,(exp_id+1)*7)]
                else:
                    test_id = [i for i in range(14,22)]

            train_id = set(total_id) - set(test_id)
            test_root = [pat_ls[i] for i in test_id]
            train_root = [pat_ls[i] for i in train_id]
            # print(train_root)

            trainset = MyDataset(args,train_root = train_root,mode='train')
            validset = MyDataset(args,train_root = test_root,mode='test')

        elif args.dataset == 'chase':
            overall_id = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14']
            test_id = overall_id[3*exp_id:3*(exp_id+1)]
            train_id = list(set(overall_id)-set(test_id))
            # print(train_id)
            trainset = MyDataset_CHASE(args,train_root = args.data_root,pat_ls=train_id,mode='train')
            validset = MyDataset_CHASE(args,train_root = args.data_root,pat_ls=test_id,mode='test')

        else:
            ####  define how you get the data on your own dataset ######
            pass

        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=6)
        val_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
        
        print("Train batch number: %i" % len(train_loader))
        print("Test batch number: %i" % len(val_loader))

        #### Above: define how you get the data on your own dataset ######
        model = DconnNet(num_class=args.num_class).cuda()

        if args.pretrained:
            model.load_state_dict(torch.load(args.pretrained,map_location = torch.device('cpu')))
            model = model.cuda()

        solver = Solver(args)

        solver.train(model, train_loader, val_loader,exp_id+1, num_epochs=args.epochs)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    
