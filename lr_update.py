import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np


def get_lr(base_lr,mode, epoch, epoch_num, gamma= None, step=None,db_size=None):


    if mode == 'step':
        lr = base_lr * (gamma ** (epoch // step))
        return lr
        
    if mode == 'poly':
        lr =base_lr * (1 - epoch / epoch_num) ** 0.9
        return lr
    if mode == 'warm-up-epoch':
        max_lr = 0.03
        lr = (1-abs((epoch+1)/(epoch_num+1)*2-1))*max_lr
        return lr

    if mode == 'warm-up-step':
        max_lr = 0.008

        niter = epoch * db_size + step
        lr, momentum = get_triangle_lr(base_lr, max_lr, epoch_num*db_size, niter, ratio=1.)
        return lr,momentum

    if mode == 'custom':
        # step_size = 20
        # ratio = 0.1
        orig_lr = 1e-5
        if epoch == 3:
        	lr = orig_lr*0.1

    
        return lr

def adjust_learning_rate(opt, optimizer, epoch):                        
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs(step = 30)
    """
    if opt.lr_mode == 'step':
        lr = opt.lr * (0.1 ** (epoch // opt.step))
    elif opt.lr_mode == 'poly':
        lr = opt.lr * (1 - epoch / opt.num_epochs) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(opt.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def get_triangle_lr(base_lr, max_lr, total_steps, cur, ratio=1., \
        annealing_decay=1e-2, momentums=[0.95, 0.85]):
    first = int(total_steps*ratio)
    last  = total_steps - first
    min_lr = base_lr * annealing_decay

    cycle = np.floor(1 + cur/total_steps)
    x = np.abs(cur*2.0/total_steps - 2.0*cycle + 1)
    if cur < first:
        lr = base_lr + (max_lr - base_lr) * np.maximum(0., 1.0 - x)
    else:
        lr = ((base_lr - min_lr)*cur + min_lr*first - base_lr*total_steps)/(first - total_steps)
    if isinstance(momentums, int):
        momentum = momentums
    else:
        if cur < first:
            momentum = momentums[0] + (momentums[1] - momentums[0]) * np.maximum(0., 1.-x)
        else:
            momentum = momentums[0]

    return lr, momentum
