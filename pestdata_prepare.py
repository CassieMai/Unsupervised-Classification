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


class  PestSubset(data.Dataset):
    def __init__(self, subset_file, root=MyPath.db_root_dir('pest'), split='train', 
                    transform=None):
        super(PestSubset, self).__init__()

        self.root = os.path.join(root, '%s' %(split))
        self.transform = transform
        self.split = split

        self.patch = True


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
            file = os.path.join(self.root, class_names[i], subdir)
            if self.patch:
                patch_paths = self.superpixels(file, class_names[i])

                for patchp in patch_paths:
                    imgs.append((patchp, class_names[i]))
            else:
                imgs.append((file, class_names[i]))
        self.imgs = imgs 
        self.classes = class_names
        print('imgs', self.imgs)
    
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


def patch_image(img_name, segs, img, label, patch_dir):
    # segs: torch.Size([B, 1, 320, 320])
    # img: torch.Size([B, 1, 3, 320, 320])
    flag = None

    # visualize images and superpixels
    if True:
        from utils.utils_xcmai import vis_superpixels
        vis_superpixels(img, segs, label)

    # patches
    if False:
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        ax.imshow(segments)
    p = 0
    patch_paths = []
    for region in regionprops(segs + 1):
        minr, minc, maxr, maxc = region.bbox
        if False:
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, 
                                fill = False, edgecolor = 'red', linewidth=2)
            ax.add_patch(rect)
        # patches
        patch = img[minr : maxr, minc : maxc, :]
        patch = cv2.resize(patch, (64, 64), interpolation=cv2.INTER_CUBIC)

        # save patches
        patch_path = os.path.join(patch_dir, img_name[:-4] + '_p' + str(p)+ '_'+ str(label)+'.jpg')
        cv2.imwrite(patch_path, patch)
        patch_paths.append(patch_path)
        p += 1
    if False:
        fig.savefig('visualization/patches ' + 'class' + str(label[i]) + 'patches' + str(np.max(segments)+1) +'.png')
        plt.close()

    flag = None
    return patch_paths

def superpixels(img_path, label, patch_dir):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, (320, 320), interpolation=cv2.INTER_CUBIC)

    num_superpix = 16
    num_levels = 5

    seeds = cv2.ximgproc.createSuperpixelSEEDS(img.shape[1], img.shape[0], img.shape[2] , num_superpix, num_levels)  # double_step = True
    seeds.iterate(img, 10)
    mask_seeds = seeds.getLabelContourMask()
    label_seeds = seeds.getLabels()
    number_seeds = seeds.getNumberOfSuperpixels()
    mask_inv_seeds = cv2.bitwise_not(mask_seeds)
    img_seeds = cv2.bitwise_and(img, img, mask = mask_inv_seeds)
    cv2.imwrite('superpixels.png', img_seeds)
    patch_paths = patch_image(img_path.split('/')[-1], label_seeds, img, label, patch_dir)
    return patch_paths
    

if __name__ == "__main__":

    # Read the subset of classes to include (sorted)
    dataset =  'pest_val.txt'   # 'debugdataset.txt'  # 'pest_train_ratio_0.4.txt'
    subset_file = '/home/xiaocmai/scratch/Unsupervised-Classification/data/pestdata_subsets/' + dataset
    root = '/home/xiaocmai/scratch/datasets/pest-classification/pest-classification/' + 'val'  # + 'train'
    with open(subset_file, 'r') as f:
        result = f.read().splitlines()
    subdirs, class_names = [], []
    for line in result:
        subdir, class_name = line.split(' ', 1)
        subdirs.append(subdir)
        class_names.append(class_name)

    # Superpixels and Gather the files (sorted)
    patch_dir = 'data/pestdata_subsets/' + dataset[:-4] + '_patches'
    if os.path.exists(patch_dir):
        shutil.rmtree(patch_dir)
    os.mkdir(patch_dir)

    newlines = []
    for i, subdir in enumerate(subdirs):
        file = os.path.join(root, class_names[i], subdir)
        patch_paths = superpixels(file, class_names[i], patch_dir)

        for p in patch_paths:
            newlines.append(p + ' ' + class_names[i] + '\n')

    with open(subset_file[:-4] + '_patches.txt', 'w') as ff:
        ff.writelines(newlines) 
