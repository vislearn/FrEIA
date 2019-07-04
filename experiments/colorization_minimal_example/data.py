import numpy as np
from skimage import io, color
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T

batch_size = 128
offsets = (47.5, 2.4, 7.4)
scales  = (25.6, 11.2, 16.8)

def norm_lab_to_rgb(L, ab, norm=True):
    '''given an Nx1xWxH Tensor L and an Nx2xwxh Tensor ab, normalized accoring to offsets and
    scales above, upsample the ab channels and combine with L, and form an RGB image.

    norm:   If false, assume that L, ab are not normalized and already in the correct range'''

    lab = torch.cat([L, ab], dim=1)
    for i in range(1 + 2*norm):
        lab[:, i] = lab[:, i] * scales[i] + offsets[i]

    lab[:, 0].clamp_(0., 100.)
    lab[:, 1:].clamp_(-128, 128)

    lab = lab.cpu().data.numpy()
    rgb = [color.lab2rgb(np.transpose(l, (1, 2, 0))).transpose(2, 0, 1) for l in lab]
    return np.array(rgb)

class LabColorDataset(Dataset):
    def __init__(self, file_list, transform=None, noise=False):

        self.files = file_list
        self.transform = transform
        self.to_tensor = T.ToTensor()
        self.noise = noise

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        im = Image.open(self.files[idx])
        if self.transform:
            im = self.transform(im)
        im = self.to_tensor(im).numpy()

        im = np.transpose(im, (1,2,0))
        if im.shape[2] != 3:
            im = np.stack([im[:,:,0]]*3, axis=2)
        im = color.rgb2lab(im).transpose((2, 0, 1))

        for i in range(3):
            im[i] = (im[i] - offsets[i]) / scales[i]
        im = torch.Tensor(im)
        if self.noise:
            im += 0.005 * torch.rand_like(im)
        return im

transf = T.Resize(64)

test_list =  [f'./train_data_128/{i}.jpg' for i in range(1, 1025)]
val_list =   [f'./train_data_128/{i}.jpg' for i in range(1025, 2049)]
train_list = [f'./train_data_128/{i}.jpg' for i in range(2049, 3033042)]

train_data = LabColorDataset(train_list, transf, noise=True)
test_data  = LabColorDataset(test_list,  transf)
val_data  =  LabColorDataset(val_list,  transf)
test_all = torch.stack(list(test_data), 0).cuda()
val_all = torch.stack(list(test_data), 0).cuda()

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8,  pin_memory=True, drop_last=True)
test_loader = DataLoader(test_data,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
