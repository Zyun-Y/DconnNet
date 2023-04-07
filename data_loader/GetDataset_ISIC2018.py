import os
import PIL
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from os import listdir
from os.path import join
from PIL import Image
import torchvision.transforms as ts
from torch.utils.data.dataset import Dataset
import random
import numbers
import torchvision.transforms.functional as TF
def randomflip_rotate(img, lab, p=0.5, degrees=0):
    if random.random() < p:
        img = TF.hflip(img)
        lab = TF.hflip(lab)
    if random.random() < p:
        img = TF.vflip(img)
        lab = TF.vflip(lab)

    if isinstance(degrees, numbers.Number):
        if degrees < 0:
            raise ValueError("If degrees is a single number, it must be positive.")
        degrees = (-degrees, degrees)
    else:
        if len(degrees) != 2:
            raise ValueError("If degrees is a sequence, it must be of len 2.")
        degrees = degrees
    angle = random.uniform(degrees[0], degrees[1])
    img = TF.rotate(img, angle)
    lab = TF.rotate(lab, angle)

    return img, lab

def ISIC2018_transform_newdata(sample, train_type):
    image, label = Image.fromarray(np.uint8(sample['image']), mode='RGB'),\
                   Image.fromarray(np.uint8(sample['label']), mode='L')

    if train_type == 'train':
        # image, label = randomcrop(size=(224, 320))(image, label)
        image, label = randomflip_rotate(image, label, p=0.5, degrees=30)
    else:
        image = image
        label = label
        
    image = ts.Compose([ts.ToTensor(),
                        ts.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])(image)
    label = ts.ToTensor()(label)

    return {'image': image, 'label': label}

class ISIC2018_dataset(Dataset):
    def __init__(self, dataset_folder='/ISIC2018_Task1_npy_all',
                 folder='folder0', train_type='train', with_name=False, transform=ISIC2018_transform_newdata):
        folder='folder'+str(folder)
        self.transform = transform
        self.train_type = train_type
        self.with_name = with_name
        self.folder_file = './data_loader/isic_datalist/' + folder

        if self.train_type in ['train', 'validation', 'test']:
            # this is for cross validation
            with open(join(self.folder_file, self.folder_file.split('/')[-1] + '_' + self.train_type + '.list'),
                      'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.folder = [join(dataset_folder, 'image', x) for x in self.image_list]
            self.mask = [join(dataset_folder, 'label', x.split('.')[0] + '_segmentation.npy') for x in self.image_list]
            # self.folder = sorted([join(dataset_folder, self.train_type, 'image', x) for x in
            #                       listdir(join(dataset_folder, self.train_type, 'image'))])
            # self.mask = sorted([join(dataset_folder, self.train_type, 'label', x) for x in
            #                     listdir(join(dataset_folder, self.train_type, 'label'))])
        else:
            print("Choosing type error, You have to choose the loading data type including: train, validation, test")

        assert len(self.folder) == len(self.mask)

    def __getitem__(self, item: int):
        image = np.load(self.folder[item])
        label = np.load(self.mask[item])
        name = self.folder[item].split('/')[-1]

        sample = {'image': image, 'label': label}

        if self.transform is not None:
            # TODO: transformation to argument datasets
            sample = self.transform(sample, self.train_type)
        
        # print(sample['label'].max())
        sample['label'] = torch.where(sample['label']>0.5,1,0)
        # conn = connectivity_matrix(sample['label'])
        # print(conn.shape)
        if self.with_name:
            return name, sample['image'], sample['label'] #,conn   
        else:
            return sample['image'], sample['label']#,conn

    def __len__(self):
        return len(self.folder)
        
    

# a = ISIC2018_dataset()



def connectivity_matrix(mask):
    # print(mask.shape)
    mask = mask.unsqueeze(1)
    [batch,channels,rows, cols] = mask.shape

    conn = np.ones([batch,8,rows, cols])
    up = np.zeros([batch,rows, cols])#move the orignal mask to up
    down = np.zeros([batch,rows, cols])
    left = np.zeros([batch,rows, cols])
    right = np.zeros([batch,rows, cols])
    up_left = np.zeros([batch,rows, cols])
    up_right = np.zeros([batch,rows, cols])
    down_left = np.zeros([batch,rows, cols])
    down_right = np.zeros([batch,rows, cols])


    up[:,:rows-1, :] = mask[:,0,1:rows,:]
    down[:,1:rows,:] = mask[:,0,0:rows-1,:]
    left[:,:,:cols-1] = mask[:,0,:,1:cols]
    right[:,:,1:cols] = mask[:,0,:,:cols-1]
    up_left[:,0:rows-1,0:cols-1] = mask[:,0,1:rows,1:cols]
    up_right[:,0:rows-1,1:cols] = mask[:,0,1:rows,0:cols-1]
    down_left[:,1:rows,0:cols-1] = mask[:,0,0:rows-1,1:cols]
    down_right[:,1:rows,1:cols] = mask[:,0,0:rows-1,0:cols-1]

    # print(mask.shape,down_right.shape)
    conn[:,0] = mask[:,0]*down_right
    conn[:,1] = mask[:,0]*down
    conn[:,2] = mask[:,0]*down_left
    conn[:,3] = mask[:,0]*right
    conn[:,4] = mask[:,0]*left
    conn[:,5] = mask[:,0]*up_right
    conn[:,6] = mask[:,0]*up
    conn[:,7] = mask[:,0]*up_left
    conn = conn.astype(np.float32)

    return conn.squeeze()

