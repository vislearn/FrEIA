import sys
import glob
from os.path import join
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import torchvision.transforms as T
from tqdm import tqdm

import joint_bilateral_filter as jbf
import config as c

offsets = (47.5, 2.4, 7.4)
scales  = (25.6, 11.2, 16.8)

def apply_filt(args):
    '''multiprocessing wrapper for applying the joint bilateral filter'''
    L_i, ab_i = args
    return jbf.upsample(L_i[0], ab_i, s_x=6, s_l=0.10)

def norm_lab_to_rgb(L, ab, norm=True, filt=False, bw=False):
    '''given an Nx1xWxH Tensor L and an Nx2xwxh Tensor ab, normalized accoring to offsets and
    scales above, upsample the ab channels and combine with L, and form an RGB image.

    norm:   If false, assume that L, ab are not normalized and already in the correct range
    filt:   Use joint bilateral upsamling to do the upsampling. Slow, but improves image quality.
    bw:     Simply produce a grayscale RGB, ignoring the ab channels'''

    if bw:
        filt=False

    if filt:
        with Pool(12) as p:
            ab_up_list = p.map(apply_filt, [(L[i], ab[i]) for i in range(len(L))])

        ab = np.stack(ab_up_list, axis=0)
        ab = torch.Tensor(ab)
    else:
        ab  = F.interpolate(ab, size=L.shape[2], mode='bilinear')

    lab = torch.cat([L, ab], dim=1)

    for i in range(1 + 2*norm):
        lab[:, i] = lab[:, i] * scales[i] + offsets[i]

    lab[:, 0].clamp_(0., 100.)
    lab[:, 1:].clamp_(-128, 128)
    if bw:
        lab[:, 1:].zero_()

    lab = lab.cpu().data.numpy()
    rgb = [color.lab2rgb(np.transpose(l, (1, 2, 0))).transpose(2, 0, 1) for l in lab]
    return np.array(rgb)

class LabColorDataset(Dataset):
    def __init__(self, file_list, transform=None):

        self.files = file_list
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        im = Image.open(self.files[idx])
        if self.transform:
            im = self.transform(im)
        im = self.to_tensor(im).numpy()

        try:
            if im.shape[0] == 1:
                im = np.concatenate([im]*3, axis=0)
            if im.shape[0] == 4:
                im = im[:3]

            im = np.transpose(im, (1,2,0))
            im = color.rgb2lab(im).transpose((2, 0, 1))
            for i in range(3):
                im[i] = (im[i] - offsets[i]) / scales[i]
            return torch.Tensor(im)

        except:
            return self.__getitem__(idx+1)


# Data transforms for training and test/validation set
transf =      T.Compose([T.RandomHorizontalFlip(),
                         T.RandomResizedCrop(c.img_dims_orig[0], scale=(0.2, 1.))])
transf_test = T.Compose([T.Resize(c.img_dims_orig[0]),
                         T.CenterCrop(c.img_dims_orig[0])])

if c.dataset == 'imagenet':
    with open('./imagenet/training_images.txt') as f:
        train_list = [join('./imagenet', fname[2:]) for fname in f.read().splitlines()]
    with open(c.validation_images) as f:
        test_list = [ t for t in f.read().splitlines()if t[0] != '#']
        test_list = [join('./imagenet', fname) for fname in test_list]
        if c.val_start is not None:
            test_list = test_list[c.val_start:c.val_stop]
else:
    data_dir = '/home/diz/data/coco17'
    complete_list = sorted(glob.glob(join(data_dir, '*.jpg')))
    train_list = complete_list[64:]
    test_list = complete_list[64:]


train_data = LabColorDataset(train_list,transf)
test_data  = LabColorDataset(test_list, transf_test)

train_loader = DataLoader(train_data, batch_size=c.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_data,  batch_size=min(64, len(test_list)), shuffle=c.shuffle_val, num_workers=4, pin_memory=True, drop_last=False)

if __name__ == '__main__':
    # Determine mean and standard deviation of RGB channels
    # (i.e. set global variables scale and offsets to 1., then use the results as new scale and offset)

    for x in test_loader:
        x_l, x_ab, _, x_ab_pred = model.prepare_batch(x)
        #continue
        img_gt = norm_lab_to_rgb(x_l, x_ab)
        img_pred = norm_lab_to_rgb(x_l, x_ab_pred)
        for i in range(c.batch_size):
            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(img_gt[i].transpose(1,2,0))
            plt.subplot(2,2,2)
            plt.scatter(x_ab[i, 0].cpu().numpy().flatten() * scales[1] + offsets[1],
                        x_ab[i, 1].cpu().numpy().flatten() * scales[2] + offsets[2], label='gt')

            plt.scatter(x_ab_pred[i, 0].cpu().numpy().flatten() * scales[1] + offsets[1],
                        x_ab_pred[i, 1].cpu().numpy().flatten() * scales[2] + offsets[2], label='pred')

            plt.legend()
            plt.subplot(2,2,3)
            plt.imshow(img_pred[i].transpose(1,2,0))

    plt.show()
    sys.exit()

    means = []
    stds = []

    for i, x in enumerate(train_loader):
        print('\r', '%i / %i' % (i, len(train_loader)), end='')
        mean = []
        std = []
        for i in range(3):
            mean.append(x[:, i].mean().item())
            std.append(x[:, i].std().item())

        means.append(mean)
        stds.append(std)

        if i >= 1000:
            break

    means, stds = np.array(means), np.array(stds)

    print()
    print('Mean   ', means.mean(axis=0))
    print('Std dev', stds.mean(axis=0))

    #[-0.04959071  0.03768991  0.11539354]
    #[0.51175581 0.17507738 0.26179135]
