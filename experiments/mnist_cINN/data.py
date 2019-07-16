import os
from os.path import join, isfile, basename
from time import time
from multiprocessing import Process

from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T

import config as c
import torchvision.datasets

def unnormalize(x):
    return x * c.data_std + c.data_mean

if c.colorize:
    data_dir = 'color_mnist_data'

    ims    = (torch.load(join(data_dir, 'color_mnist_images.pt')) - c.data_mean) / c.data_std
    labels = torch.load(join(data_dir, 'color_mnist_labels.pt'))
    masks  = torch.load(join(data_dir, 'color_mnist_masks.pt'))

    dataset = torch.utils.data.TensorDataset(ims, labels, masks)

    train_loader  = DataLoader(dataset, batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader   = train_loader

else:
    data_dir = 'mnist_data'

    train_data = torchvision.datasets.MNIST(data_dir, train=True, transform=T.ToTensor(), download=True)
    test_data = torchvision.datasets.MNIST(data_dir, train=False, transform=T.ToTensor(), download=True)

    train_loader  = DataLoader(train_data, batch_size=c.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    test_loader   = DataLoader(test_data,  batch_size=c.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
