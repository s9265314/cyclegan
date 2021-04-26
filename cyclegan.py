#%%
# -*- coding: utf-8 -*-
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import itertools
from torch.autograd import Variable
from PIL import Image
import glob
import os
import cv2
import random
Tensor = torch.cuda.FloatTensor
from data_loader import ImageDataset,save_img10,show_img
from model import Discriminator,Generator,ResidualBlock,ReplayBuffer

#%%

# input_A = Tensor(batchSize,3,256,256)
# input_B = Tensor(batchSize,3,256,256)
# if 1:
#     for times, batch in enumerate(dataloader):
#         times += 1
#         td = Variable(input_A.copy_(batch['d']))
#         tc = Variable(input_B.copy_(batch['c']))
#         if times == 1:
#             break 

#%%


# #%%
# def t2n(x):
#     return  x.detach().numpy()
# #tensor2numpy
# cf = torch.squeeze(Gd2c(td)).cpu()
# cf=cf
# df = torch.squeeze(Gc2d(tc))
# df=df.cpu()
# td = torch.squeeze(td.cpu())
# tc = torch.squeeze(tc.cpu())
# td = t2n(td)
# tc = t2n(tc)
# df = t2n(df)
# cf = t2n(cf)
# #%%
# imgs = [td,df,tc,cf]
# title = ['d', 'df', 'c', 'cf']
# contrast = 8
# for i in range(len(imgs)):
#   plt.subplot(2, 2, i+1)
#   plt.title(title[i])
#   if i % 2 == 0:
#     plt.imshow(imgs[i][0])
#   else:
#     plt.imshow(imgs[i][0])
# plt.show()


#%%

# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Model
Gd2c = Generator(3,3).to(device)
Gc2d = Generator(3,3).to(device)
Dd = Discriminator(3).to(device)
Dc = Discriminator(3).to(device)
# Gd2c.load_state_dict(torch.load("./output/Gd2c_250.pth"))
# Gc2d.load_state_dict(torch.load("./output/Gc2d_250.pth"))
# Dd.load_state_dict(torch.load("./output/Dd_250.pth"))
# Dc.load_state_dict(torch.load("./output/Dc_250.pth"))
print(Gd2c)
print(Dd)

# Settings
lr = 0.0002
epoch = 0
n_epochs=200000
decay_epoch=100
size = 256
batchSize = 3

input_A = Tensor(batchSize,3,256,256)
input_B = Tensor(batchSize,3,256,256)
target_real = torch.ones(batchSize,1,requires_grad =False).to(device)
target_fake = torch.zeros(batchSize,1,requires_grad =False).to(device)

# Load data

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(Gd2c.parameters(), Gc2d.parameters()),
                                lr=0.0002, betas=(0.5, 0.999))
optimizer_Dc = torch.optim.Adam(Dc.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_Dd = torch.optim.Adam(Dd.parameters(), lr=0.0002, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_Dc = torch.optim.lr_scheduler.LambdaLR(optimizer_Dc, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_Dd = torch.optim.lr_scheduler.LambdaLR(optimizer_Dd, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
#%%
size = 256
batchSize = 3

transforms_ = [ transforms.Resize(int(size*1.12)), 
                transforms.RandomCrop(size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()]

# transforms_ = [ transforms.Resize(int(size*1.12), Image.BICUBIC), 
#                 transforms.RandomCrop(size), 
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]    

dataloader = DataLoader(ImageDataset('dataset/', transforms_=transforms_, unaligned=True), 
                        batch_size=batchSize, shuffle=True,drop_last = True)
#%%
# Train
path = './output/h-z/'
for epoch in range(epoch,n_epochs):
    epoch += 1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tStart = time.time()
    for times, batch in enumerate(dataloader):
        d = Variable(input_A.copy_(batch['d']))
        c = Variable(input_B.copy_(batch['c']))
        optimizer_G.zero_grad()
        #generator_img
        #with torch.no_grad():
        cf = Gd2c(d)
        df = Gc2d(c)
        cr = Gd2c(df)
        dr = Gc2d(cf)

        # identity loss
        Gd2c_identity_loss = criterion_identity(Gd2c(c),c)
        Gc2d_identity_loss = criterion_identity(Gc2d(d),d)
        identity_loss = (Gc2d_identity_loss + Gd2c_identity_loss)*0.5
        #generator_loss
        Gd2c_loss = criterion_GAN(Dc(cf), target_real)
        Gc2d_loss = criterion_GAN(Dd(df), target_real)
        G_loss = (Gc2d_loss + Gd2c_loss)*0.5
        #cycle_loss
        cdc_cycle_loss = criterion_cycle(cr,c)
        dcd_cycle_loss = criterion_cycle(dr,d)
        cycle_loss = (dcd_cycle_loss + cdc_cycle_loss)*0.5
        #total_loss
        loss_G = cycle_loss*10 + G_loss + identity_loss*0.5
        loss_G.backward()
        optimizer_G.step()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        optimizer_Dc.zero_grad()
        #real
        Dc_c_loss = criterion_GAN(Dc(c),target_real)
        #fake
        cf = ReplayBuffer().push_and_pop(cf)
        Dc_cf= Dc(cf.detach())
        Dc_cf_loss = criterion_GAN(Dc_cf,target_fake)
        # Total loss
        loss_Dc = (Dc_c_loss + Dc_cf_loss)*0.5
        loss_Dc.backward()

        optimizer_Dc.step()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        optimizer_Dd.zero_grad()
        #real
        Dd_d_loss = criterion_GAN(Dd(d),target_real)
        #fake
        df = ReplayBuffer().push_and_pop(df)
        Dd_df= Dd(df.detach())
        Dd_df_loss = criterion_GAN(Dd_df,target_fake)
        # Total loss
        loss_Dd = (Dd_d_loss + Dd_df_loss)*0.5
        loss_Dd.backward()
        
        optimizer_Dd.step()

        
        if times %  20 == 0:
            print('times:',times,'Loss_G:',loss_G.item(),'loss_Dd:' ,loss_Dd.item(), "loss_Dc:" ,loss_Dc.item(), end='\n')
    # Update learning rates
    cost_t = time.time() -tStart
    lr_scheduler_G.step(epoch)
    lr_scheduler_Dc.step(epoch)
    lr_scheduler_Dd.step(epoch)
    save_img10(c,"c_real",path,epoch)
    save_img10(d,"d_real",path,epoch)
    save_img10(cf,"c_fake",path,epoch)
    save_img10(df,"d_fake",path,epoch)
    if epoch % 10 == 0:
        torch.save(Gc2d.state_dict(), './output/Gc2d_'+str(epoch)+'.pth')
        torch.save(Gd2c.state_dict(), './output/Gd2c_'+str(epoch)+'.pth')
        torch.save(Dc.state_dict(), './output/Dc_'+str(epoch)+'.pth')
        torch.save(Dd.state_dict(), './output/Dd_'+str(epoch)+'.pth')
    
    print('Epoch = ',epoch+1,"/",n_epochs,' Loss_G:',loss_G.item(),'loss_Dd:' ,loss_Dd.item(), "loss_Dc:" ,loss_Dc.item(),"times:",cost_t, end='')


# %%
