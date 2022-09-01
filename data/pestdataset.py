"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
from PIL import Image
from utils.mypath import MyPath
from torchvision import transforms as tf
from glob import glob
import cv2
import numpy as np
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import os
import shutil


class ImageNet(datasets.ImageFolder):
    def __init__(self, root=MyPath.db_root_dir('imagenet'), split='train', transform=None):
        super(ImageNet, self).__init__(root=os.path.join(root, 'ILSVRC2012_img_%s' %(split)),
                                         transform=None)
        self.transform = transform 
        self.split = split
        self.resize = tf.Resize(256)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img


class  PestSubset(data.Dataset):
    def __init__(self, subset_file, root=MyPath.db_root_dir('pest'), split='train', 
                    transform=None):
        super(PestSubset, self).__init__()

        if split == 'train':
            self.root = root 
        else:
            self.root = os.path.join(root, '%s' %(split))
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            if split == 'train':
                file = os.path.join(self.root, subdir)
            else:
                file = os.path.join(self.root, class_names[i], subdir)
            imgs.append((file, class_names[i]))
        self.imgs = imgs 
        self.classes = class_names
    
	    # Resize
        self.resize = tf.Resize(256)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img) 
        class_name = target

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}
        # print('out', out)
        return out
