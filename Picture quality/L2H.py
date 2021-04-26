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
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.transform256 = transforms.Compose(transforms_256)
        self.transform128 = transforms.Compose(transforms_128)
        self.transform64 = transforms.Compose(transforms_64)
        self.unaligned = unaligned
        self.files_d = sorted(glob.glob(os.path.join(root) + '/*.*'))

    def __getitem__(self, index):
        item = self.transform(Image.open(self.files_d[index % len(self.files_d)]))
        item_256 = self.transform256(item)
        item_128 = self.transform128(item)
        item_64 = self.transform64(item)
        #return {'d': item_d, 'c': item_c}
        return {'h': item_256,'m':item_128,'l':item_64}

    def __len__(self):
        return len(self.files_d)
class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))
#%%

save_path = './L2H_intput'
size = 256
batchSize = 10
transforms_ = [ transforms.Resize(int(size*1.12)), 
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(size)]
transforms_256 = [transforms.ToTensor()]
transforms_128 = [transforms.Resize(int(size/2)),
                  transforms.ToTensor()]
transforms_64 = [transforms.Resize(int(size/4)),
                  transforms.ToTensor()]

dataloader = DataLoader(ImageDataset(save_path, transforms_=transforms_, unaligned=True), 
                        batch_size=batchSize, shuffle=True,drop_last=True)
#%%
files_d = sorted(glob.glob(os.path.join(save_path) + '/*.*'))
im = Image.open(files_d[0])
#%%
input_H = Tensor(batchSize,3,256,256)
input_M = Tensor(batchSize,3,128,128)
input_L = Tensor(batchSize,3,64,64)
#%%
for epoch in range(0,1):
    epoch += 1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for times, batch in enumerate(dataloader):
        h = Variable(input_H.copy_(batch['h']))
        m = Variable(input_M.copy_(batch['m']))
        l = Variable(input_L.copy_(batch['l']))
        break
    break
#%%
def tensor_to_np(tensor):
    img = tensor
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img
def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
def show_img(x):
    a=torch.chunk(x,x.shape[0], dim=0)
    for i in range(x.shape[0]):
        show_from_tensor(a[i])
#%%
# Discriminator Loss => BCELoss
def d_loss_function(inputs, targets):
    return nn.BCELoss()(inputs, targets)


def g_loss_function(inputs):
    targets = torch.ones([inputs.shape[0], 1])
    targets = targets.to(device)
    return nn.BCELoss()(inputs, targets)
#%%
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
class half_img(nn.Module):
    def __init__(self):
        super(half_img, self).__init__()
        # Initial convolution block       
        model = [  nn.AvgPool2d(2) ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, half, double, n_residual_blocks=7):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]
        #64*256*256
        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(half):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2
        #256*64*64
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(double):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2
        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    nn.Tanh() ]
        #3*256*256
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)
#%%
# GPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('GPU State:', device)

# Model
Gl2m = Generator(3,3,1,2).cuda()
Gm2h = Generator(3,3,1,2).cuda()
Dm = Discriminator(3).to(device)
Dh = Discriminator(3).to(device)

Gl2m.load_state_dict(torch.load("./model/Gl_250.pth"))
Gm2h.load_state_dict(torch.load("./model/Gh_250.pth"))
Dm.load_state_dict(torch.load("./model/Dl_250.pth"))
Dh.load_state_dict(torch.load("./model/Dh_250.pth"))

h2m = half_img().to(device)
m2l = half_img().to(device)
# Settings
lr = 0.0002
epoch = 0
n_epochs=200000
decay_epoch=100

input_H = Tensor(batchSize,3,256,256)
input_M = Tensor(batchSize,3,128,128)
input_L = Tensor(batchSize,3,64,64)
target_real = Variable(Tensor(batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(batchSize).fill_(0.0), requires_grad=False)
#%%
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
#%%
# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()
# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(Gl2m.parameters(), Gm2h.parameters()),
                                lr=0.0002, betas=(0.5, 0.999))
optimizer_Dm = torch.optim.Adam(Dm.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_Dh = torch.optim.Adam(Dh.parameters(), lr=0.0002, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_Dm = torch.optim.lr_scheduler.LambdaLR(optimizer_Dm, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)
lr_scheduler_Dh = torch.optim.lr_scheduler.LambdaLR(optimizer_Dh, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step)

#%%
# Train
path = './L2H_output/'
epoch = 250
for epoch in range(epoch,n_epochs):
    epoch += 1
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tStart = time.time()
    for times, batch in enumerate(dataloader):
        H = Variable(input_H.copy_(batch['h']))
        M = Variable(input_M.copy_(batch['m']))
        L = Variable(input_L.copy_(batch['l']))
        optimizer_G.zero_grad()
        #generator_img
        #with torch.no_grad():
        loss_Dm = 0
        loss_Dh = 0
        m_1 = Gl2m(L)
        l_2 = m2l(m_1)
        
        h_1 = Gm2h(M)
        m_2 = h2m(h_1)
        h_2 = Gm2h(m_1)
        m_3 = h2m(h_2)
        #cycle_loss(L1)
        Gl2m_cycle_loss = criterion_cycle(l_2,L)
        Gm2h_cycle_loss = criterion_cycle(m_2,M)+(criterion_cycle(m_3,m_1)*0.5)
        cycle_loss = (Gl2m_cycle_loss + Gm2h_cycle_loss*0.67)*0.5
        
        #generator_loss(MSE)
        Gl2m_loss = criterion_GAN(Dm(m_1), target_real)
        Gm2h_loss = criterion_GAN(Dh(h_1), target_real)+(criterion_GAN(Dh(h_2), target_real)*0.5)
        G_loss = (Gl2m_loss + Gm2h_loss*0.67)*0.5
        
        # identity loss(L1)
        Gl2m_id_loss = criterion_identity(m_1,M)
        Gm2h_id_loss = criterion_identity(h_1,H)+(criterion_identity(h_2,H)*0.5)
        identity_loss = (Gl2m_id_loss + Gm2h_id_loss*0.67)*0.5

        #total_loss
        loss_G = identity_loss*0.5+G_loss+cycle_loss*10
        loss_G.backward()
        optimizer_G.step()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        optimizer_Dm.zero_grad()
        #real
        Dm_loss = criterion_GAN(Dm(M),target_real)
        #fake
        m_1 = ReplayBuffer().push_and_pop(m_1)
        m_2 = ReplayBuffer().push_and_pop(m_2)
        m_3 = ReplayBuffer().push_and_pop(m_3)
        Dm_f_loss = criterion_GAN(Dm(m_1.detach()),target_fake)
        Dm_f_loss += criterion_GAN(Dm(m_2.detach()),target_fake)*0.5
        Dm_f_loss += criterion_GAN(Dm(m_3.detach()),target_fake)*0.25
        # Total loss
        loss_Dm = (Dm_f_loss+Dm_loss)*0.5
        
        loss_Dm.backward()
        optimizer_Dm.step()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
        optimizer_Dh.zero_grad()
        #real
        Dh_loss = criterion_GAN(Dh(H),target_real)
        #fake
        h_1 = ReplayBuffer().push_and_pop(h_1)
        h_2 = ReplayBuffer().push_and_pop(h_2)

        Dh_f_loss = criterion_GAN(Dh(h_1.detach()),target_fake)+criterion_GAN(Dh(h_2.detach()),target_fake)
        # Total loss
        loss_Dh = (Dh_f_loss*0.5+Dh_loss)*0.5
        loss_Dh.backward()

        optimizer_Dh.step()

        
        if times %  20 == 0:
            print('times:',times,'Loss_G:',loss_G.item(),'loss_Dm:' ,loss_Dm.item(), "loss_Dh:" ,loss_Dh.item(), end='\n')
    # Update learning rates
    cost_t = time.time() -tStart
    lr_scheduler_G.step(epoch)
    lr_scheduler_Dm.step(epoch)
    lr_scheduler_Dh.step(epoch)
    save_img10(M,"M_real",path,epoch)
    save_img10(H,"H_real",path,epoch)
    save_img10(h_1,"h_1",path,epoch)
    save_img10(h_2,"h_2",path,epoch)
    save_img10(m_1,"m_1",path,epoch)
    save_img10(m_1,"m_2",path,epoch)
    save_img10(m_1,"m_3",path,epoch)
    if epoch % 10 == 0:
        torch.save(Gl2m.state_dict(), './model/Gl_'+str(epoch)+'.pth')
        torch.save(Gm2h.state_dict(), './model/Gh_'+str(epoch)+'.pth')
        torch.save(Dm.state_dict(), './model/Dl_'+str(epoch)+'.pth')
        torch.save(Dh.state_dict(), './model/Dh_'+str(epoch)+'.pth')
    
    print('Epoch = ',epoch+1,"/",n_epochs,' Loss_G:',loss_G.item(),'loss_Dm:' ,loss_Dm.item(), "loss_Dh:" ,loss_Dh.item(),"times:",cost_t, end='')