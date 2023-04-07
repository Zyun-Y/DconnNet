# -*- coding: UTF-8 -*-

from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
import glob
import os
import scipy.io as scio
from skimage.io import imread, imsave
import numpy as np
import torch.nn.functional as F
import cv2

class Normalize(object):
    def __call__(self, image, mask=None):
        # image = (image - self.mean)/self.std
        image = (image-image.min())/(image.max()-image.min())
        if mask is None:
            return image
        return image, mask

class RandomCrop(object):
    def __call__(self, image, mask=None):
        H,W   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        if mask is None:
            return image[p0:p1,p2:p3, :]
        return image[p0:p1,p2:p3], mask[p0:p1,p2:p3]

class RandomFlip(object):
    def __call__(self, image, mask=None):
        if np.random.randint(2)==0:
            if mask is None:
                return image[:,::-1,:].copy()
            return image[:,::-1].copy(), mask[:, ::-1].copy()
        else:
            if mask is None:
                return image
            return image, mask

def Resize(image, mask,H,W):
    image = cv2.resize(image, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
    if mask is not None:
        mask  = cv2.resize( mask, dsize=(W, H), interpolation=cv2.INTER_LINEAR)
        return image, mask
    else:
        return image

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = torch.from_numpy(image)
        if mask is None:
            return image
        mask  = torch.from_numpy(mask)

        return image, mask

def _resize_image(image, target):
   return cv2.resize(image, dsize=(target[0], target[1]), interpolation=cv2.INTER_LINEAR)


class MyDataset(data.Dataset):# 
    def __init__(self, args,train_root, mode='train'): 
        self.args = args
        img_ls = []
        mask_ls = []
        name_ls = []

        self.mode = mode
        self.slice_cnt = []

        pat_cnt=0
        self.cnt= 0
        for pat in train_root:
            mask_p = pat+'/mask'
            orig_p = pat+'/orig'
            # masks = glob.glob(mask_p+'/*')
            imgs = glob.glob(orig_p+'/*')
            i=0
            for img in imgs:
                # print(img)
                name = img.split('orig/')[1]
                mask = mask_p+'/'+name
                self.cnt+=1
                
                img_ls.append(img)
                name_ls.append(name)
                mask_ls.append(mask)

                self.slice_cnt.append(pat_cnt+1)

            pat_cnt+=1

        self.name_ls = name_ls
        self.img_ls = img_ls

        self.mask_ls = mask_ls

        self.normalize  = Normalize()
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()

        self.totensor   = ToTensor()
        # print(self.slice_cnt)
    def __getitem__(self, index):

        img  = cv2.imread(self.img_ls[index]).astype(np.float32)
        mask  = cv2.imread(self.mask_ls[index],0).astype(np.float32)

        if self.mode == 'train':
            img = self.normalize(img)
            img, mask = self.randomflip(img, mask)
            img = np.transpose(img,(2,0,1))
            img = np.expand_dims(img,axis=0)
            mask = np.expand_dims(mask,axis=0)
            # generate onehot data
            # palette = [[0], [1],[2], [3]]
            
            # onehotmask = mask_to_onehot(mask, palette)
            # onehotmask = np.transpose(onehotmask,(2,0,1))

            img, mask = self.totensor(img, mask)
            # img = img.cuda()

            # print(mask.shape,img.shape)
            if self.args.num_class == 1:
                mask = F.interpolate(mask.unsqueeze(0),size=self.args.resize)
                mask = (mask>0.5).float()
            else:
                mask = F.interpolate(mask.unsqueeze(0),size=self.args.resize, mode='nearest')

            img = F.interpolate(img,size=self.args.resize)
            # img = Resize(img[0], None,512,512)

            mask = mask.squeeze()
            img = img.squeeze()


            # print(mask.shape,img.shape)
            # os._exit(0)
            return img,mask
        else:
            # print(self.slice_cnt)
            pat_id = self.slice_cnt[index]

            img, _ = self.normalize(img, mask)
            # img, mask = self.randomflip(img, mask)
            img = np.transpose(img,(2,0,1))
            img = np.expand_dims(img,axis=0)
            mask = np.expand_dims(mask,axis=0)
            # generate onehot data
            img, mask = self.totensor(img, mask)
            if self.args.num_class == 1:
                mask = F.interpolate(mask.unsqueeze(0),size=self.args.resize)
                mask = (mask>0.5).float()
                onehotmask = mask
            else:
                mask = F.interpolate(mask.unsqueeze(0),size=self.args.resize, mode='nearest')
                # print(mask.shape)
                onehotmask = F.one_hot(mask.squeeze(1).long(),4).permute(0,3,1,2)
            img = F.interpolate(img,size=self.args.resize)
            # print(img.shape,onehotmask.shape)
            img = img.squeeze()
            onehotmask = onehotmask.squeeze()
            name = self.name_ls[index]

            return img,onehotmask,name, pat_id



    def __len__(self): 
        return self.cnt




def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def check_label(mask):
    label = np.array([1,0,0,0])
    # print(mask.shape)
    # print(mask[1,:,:].max())
    if mask[1,:,:].max()!=0:
        label[1]=1

    if mask[2,:,:].max()!=0:
        label[2]=1

    if mask[3,:,:].max()!=0:
        label[3]=1

    return label

