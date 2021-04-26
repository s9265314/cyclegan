#%%
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import transforms
from torch.autograd import Variable
from PIL import Image
import glob
import os
import random
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