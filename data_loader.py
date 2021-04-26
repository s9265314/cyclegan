#%%
#%%
# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import cv2
import random
Tensor = torch.cuda.FloatTensor
#%%
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_d = sorted(glob.glob(os.path.join(root+'/h') + '/*.*'))
        self.files_c = sorted(glob.glob(os.path.join(root+'/z') + '/*.*'))

    def __getitem__(self, index):
        item_d = self.transform(Image.open(self.files_d[index % len(self.files_d)]))

        if self.unaligned:
            item_c = self.transform(Image.open(self.files_c[random.randint(0, len(self.files_c) - 1)])) #隨機採一張
        else:
            item_c = self.transform(Image.open(self.files_c[index % len(self.files_c)])) #沒潤燈

        return {'d': item_d, 'c': item_c}

    def __len__(self):
        return max(len(self.files_d), len(self.files_c))

#%%
def save_img10(batch_img,type0,path,epoch):
    n=batch_img.shape[0]
    a=torch.chunk(batch_img, n, dim=0)
    for i in range(n):
        img = a[i].clone()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        if i < 5:
            cv2.imwrite(path+str(epoch)+"-"+str(i)+"-"+type0+'.png',cv2.cvtColor(img*255, cv2.COLOR_RGB2BGR))
def show_img(batch_img):
    n=batch_img.shape[0]
    a=torch.chunk(batch_img, n, dim=0)
    for i in range(n):
        img = a[i].clone()
        img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        cv2.imshow('My Image', cv2.cvtColor(img*255, cv2.COLOR_RGB2BGR))
        # Discriminator Loss => BCELoss
