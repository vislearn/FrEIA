from glob import glob
from multiprocessing import Pool

from tqdm import tqdm
import numpy as np
#import skimage.io as io
import matplotlib.pyplot as plt

def bootstrap_mean(x):
    '''calculate mean with error estimate through statistical bootsrapping'''
    mean = np.mean(x)
    mean_sampled = []
    for i in range(256):
        x_resamp = x[np.random.randint(0, len(x), size=x.shape)]
        mean_sampled.append(np.mean(x_resamp))
    return mean, np.std(mean_sampled)

# For index i, each function returns a list of filenames of one or more (diverse) colorization results
def cinn_imgs(i):
    return ['./images/val_set/%.6i/%.2i.png' % (i, j) for j in range(2, 10)]

def vae_imgs(i):
    return ['/home/diz/code/colorization_baselines/vae_diverse_colorization/data/output/testimgs/%.6i.png/divcolor_%.3i.png' % (i, j) for j in range(8)]

def cgan_imgs(i):
    return ['/home/diz/single_images_cgan/%.6i.png' % (i)]

def cnn_imgs(i):
    return ['/home/diz/code/colorization_baselines/siggraph2016_colorization/single_images_cnn/%.6i.png' % (i)]

def gt_img(i):
    return '/home/diz/data/imagenet/val_cropped/%.6i.png' % (i)

def abl_no_imgs(i):
    return ['./images/val_set_ablation_no_cond/%.6i/%.2i.png' % (i, j) for j in range(2, 10)]

def abl_fixed_imgs(i):
    return ['./cond_fixed/imgs/%.6i_%.3i.png' % (i, j) for j in range(0, 8)]

def variance(functs):
    '''For the methods given in *functs (returning image filenames), compute the variance of
    colorizations per image, averaged over the 5k val. set'''
    for f in functs:
        var = []
        for i in tqdm(range(5120)):
            imgs = np.stack([plt.imread(fname)[:, :, :3] for fname in f(i)], axis=0)
            var.append(np.mean(np.var(imgs, axis=0)))

        print('variance', f.__name__, *bootstrap_mean(np.array(var)))

def err_individual_image(args):
    '''wrapper for multiprocessing: args is a tuple (f,i) of filename-returning-function f, and 
    val. index i.'''

    i, f = args
    gt_im = plt.imread(gt_img(i))[np.newaxis, :, :, :3]
    imgs = np.stack([plt.imread(fname)[:, :, :3] for fname in f(i)], axis=0)
    imgs_1 = np.stack([plt.imread(fname)[:, :, :3] for fname in f(i)[:1]], axis=0)

    err = np.sqrt(np.mean(np.min((imgs - gt_im)**2, axis=0)))
    err_1 = np.sqrt(np.mean((imgs_1 - gt_im)**2))

    return [err, err_1]

def err(functs):
    '''Compute the RMS best-of-8 and best-of-1 error on the 5k val set'''
    for f in functs:
        args = [(i,f) for i in range(5120)]
        with Pool(16) as p:
            errs = np.array(p.map(err_individual_image, args))

        print('of 8', f.__name__, *bootstrap_mean(errs[:,0]))
        print('of 1', f.__name__, *bootstrap_mean(errs[:,1]))

functs = [cgan_imgs, cinn_imgs, vae_imgs, cnn_imgs, abl_no_imgs, abl_fixed_imgs]
functs_diverse = [cinn_imgs, vae_imgs, abl_no_imgs, abl_fixed_imgs]

err(functs)
# of 8:
#   cinn 3.53 0.04
#   vae  4.06 0.04
#   cnn  6.77 0.05
#   cgan 9.75 0.06
#   abl no
#   abl fixed
# of 1:
#   cinn 9.52 0.06
#   vae  8.40 0.07
#   cnn  6.77 0.05
#   cgan 9.75 0.06
#   abl no
#   abl fixed

variance(functs_diverse)
# cinn  0.00352 2.71e-05
# vae   0.00210 2.13e-05
# abl no
# abl fixed
