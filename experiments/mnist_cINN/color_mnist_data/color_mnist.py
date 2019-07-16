import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from scipy.ndimage import zoom
from skimage.filters import gaussian
import torch

from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as T
import torchvision.datasets

data_dir = '../mnist_data'

hues_sigmas = {
                0: (0, 20),
                1: (50, 10),
                2: (110, 30),
                3: (180, 20),
                4: (235, 25),
                5: (305, 25),
              }

pairings = [
            (0,2),
            (1,3),
            (2,4),
            (3,5),
            (4,0),
            (5,1),
            (0,4),
            (2,0),
            (4,2),
            (5,3),
        ]

imsize = 28

def colorize(img, fg, bg):
    base_fg = hues_sigmas[fg][0] + hues_sigmas[fg][1] * np.random.randn()
    base_bg = hues_sigmas[bg][0] + hues_sigmas[bg][1] * np.random.randn()
    img_out = 0.8 * np.ones((imsize, imsize, 3))
    img_out[:, :, 0] = img * base_fg / 360.
    img_out[:, :, 0] += (1.-img) * base_bg / 360.

    noise = np.random.randn(3, imsize, imsize)
    noise[0] = 0.25 * gaussian(noise[0], 4)
    noise[1] = 0.3 * gaussian(noise[1], 2)
    noise[2] = 0.05 * noise[2]

    img_out += noise.transpose((1,2,0))
    img_out[:, :, 0] = img_out[:, :, 0] % 1.
    img_out[:, :, 1:] = np.clip(img_out[:, :, 1:], 0, 1)

    return np.clip(hsv_to_rgb(img_out), 0, 1)

train_data = torchvision.datasets.MNIST(data_dir, train=True, transform=T.ToTensor(), download=True)
test_data = torchvision.datasets.MNIST(data_dir, train=False, transform=T.ToTensor(), download=True)

train_loader  = DataLoader(train_data, batch_size=512, shuffle=False)

images = [im.numpy() for im, labels in train_loader]
labels = [labels.numpy() for im, labels in train_loader]
images = np.concatenate(images, axis=0)
labels = np.concatenate(labels, axis=0)

def export():
    from tqdm import tqdm

    imgs_color = []
    for i in tqdm(range(len(labels))):
        im_color = colorize(images[i, 0], *(pairings[labels[i]]))
        imgs_color.append(im_color.transpose((2,0,1)))

    imgs_torch = torch.Tensor(images)
    imgs_color_torch = torch.Tensor(np.stack(imgs_color, axis=0))
    labels_torch = torch.Tensor(labels)

    torch.save(imgs_color_torch, 'color_mnist_images.pt')
    torch.save(imgs_torch, 'color_mnist_masks.pt')
    torch.save(labels_torch, 'color_mnist_labels.pt')

def plot():
    n_rows = 10
    n_cols = 16

    for i in range(n_rows):
      matching_ims = images[labels == i]
      colors = pairings[i]

      for j in range(n_cols):
        im = colorize(matching_ims[j, 0], *colors)

        plt.subplot(n_rows, n_cols, n_cols*i+j+1)
        plt.imshow(im)
        plt.xticks([])
        plt.yticks([])

    plt.show()

export()
