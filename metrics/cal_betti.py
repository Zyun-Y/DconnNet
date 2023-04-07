import matplotlib

matplotlib.use('Agg')
import time
import torch
import torch.nn as nn
import os
# import visdom
import random
from tqdm import tqdm as tqdm
import sys
from betti_compute import betti_number
# from TDFMain import *
# from TDFMain_pytorch import *




def getBetti(binaryPredict, masks):
    predict_betti_number_ls = []
    groundtruth_betti_number_ls =[]
    betti_error_ls = []
    topo_size = 65
    gt_dmap = masks.cuda()
    # et_dmap = likelihoodMap_final
    # n_fix = 0
    # n_remove = 0
    # topo_cp_weight_map = np.zeros(et_dmap.shape)
    # topo_cp_ref_map = np.zeros(et_dmap.shape)
    # allWindows = 1
    # inWindows = 1

    for y in range(0, gt_dmap.shape[0], topo_size):
        for x in range(0, gt_dmap.shape[1], topo_size):
            # likelihoodAll = []
            # allWindows = allWindows + 1
            # likelihood = et_dmap[y:min(y + topo_size, gt_dmap.shape[0]),
            #              x:min(x + topo_size, gt_dmap.shape[1])]
            binary = binaryPredict[y:min(y + topo_size, gt_dmap.shape[0]),
                         x:min(x + topo_size, gt_dmap.shape[1])]           
            groundtruth = gt_dmap[y:min(y + topo_size, gt_dmap.shape[0]),
                          x:min(x + topo_size, gt_dmap.shape[1])]
            # for likelihoodMap in likelihoodMaps:
            #     likelihoodAll.append(likelihoodMap[y:min(y + topo_size, gt_dmap.shape[0]),
            #              x:min(x + topo_size, gt_dmap.shape[1])])

            # print('likelihood', likelihood.shape, 'groundtruth', groundtruth.shape, 'binaryPredict', binary.shape)
            predict_betti_number = betti_number(binary)
            groundtruth_betti_number = betti_number(groundtruth)
            # print(predict_betti_number, groundtruth_betti_number)
            predict_betti_number_ls.append(predict_betti_number)
            groundtruth_betti_number_ls.append(groundtruth_betti_number)
            betti_error_ls.append(abs(predict_betti_number-groundtruth_betti_number))

    return betti_error_ls
