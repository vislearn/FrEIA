from os.path import join

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance_matrix

import model
import data

cinn = model.ColorizationCINN(0)
cinn.cuda()
cinn.eval()
state_dict = {k:v for k,v in torch.load('output/lsun_cinn.pt').items() if 'tmp_var' not in k}
cinn.load_state_dict(state_dict)

def colorize_test_set(temp=1., postfix=0, img_folder='images'):
    '''Colorize the whole test set once.
    temp:       Sampling temperature
    postfix:    Has to be integer. Append to file name (e.g. to make 10 diverse colorizations of test set)
    '''
    counter = 0
    with torch.no_grad():
        for Lab in tqdm(data.test_loader):
            Lab = Lab.cuda()
            z = temp * torch.randn(Lab.shape[0], model.ndim_total).cuda()
            L, ab = Lab[:, :1], Lab[:, 1:]

            ab_gen = cinn.reverse_sample(z, L)
            rgb_gen = data.norm_lab_to_rgb(L.cpu(), ab_gen.cpu())

            for im in rgb_gen:
                im = np.transpose(im, (1,2,0))
                plt.imsave(join(img_folder, '%.6i_%.3i.png' % (counter, postfix)), im)
                counter += 1


def best_of_n(n):
    '''computes the best-of-n MSE metric'''
    with torch.no_grad():
        errs_batches = []
        for Lab in tqdm(data.test_loader, disable=True):
            L =  Lab[:, :1].cuda()
            ab = Lab[:, 1:].cuda()
            B = L.shape[0]

            rgb_gt = data.norm_lab_to_rgb(L.cpu(), ab.cpu())
            rgb_gt = rgb_gt.reshape(B, -1)

            errs = np.inf * np.ones(B)

            for k in range(n):
                z = torch.randn(B, model.ndim_total).cuda()
                ab_k = cinn.reverse_sample(z, L)
                rgb_k = data.norm_lab_to_rgb(L.cpu(), ab_k.cpu()).reshape(B, -1)

                errs_k = np.mean((rgb_k - rgb_gt)**2, axis=1)
                errs = np.minimum(errs, errs_k)

            errs_batches.append(np.mean(errs))

        print(F'MSE best of {n}')
        print(np.sqrt(np.mean(errs_batches)))
        return np.sqrt(np.mean(errs_batches))

def rgb_var(n):
    '''computes the pixel-wise variance of samples'''
    with torch.no_grad():
        var = []
        for Lab in tqdm(data.test_all, disable=True):
            L = Lab[:1].view(1,1,64,64).expand(n, -1, -1, -1).cuda()
            z = torch.randn(n, model.ndim_total).cuda()

            ab = cinn.reverse_sample(z, L)
            rgb = data.norm_lab_to_rgb(L.cpu(), ab.cpu()).reshape(n, -1)

            var.append(np.mean(np.var(rgb, axis=0)))

        print(F'Var (of {n} samples)')
        print(np.mean(var))
        print(F'sqrt(Var) (of {n} samples)')
        print(np.sqrt(np.mean(var)))

for i in range(8):
    torch.manual_seed(i+111)
    colorize_test_set(postfix=i)
