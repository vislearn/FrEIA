#!/usr/bin/env python
'''
Usage: ./eval.py model_checkpoint_file [val_start_index, val_stop_index]
model_checkpoint_file:          Path of the checkpoint
optional val_start/stop_index:  Only use validation images between these indexes
                                (Useful for GNU-parallel etc.)
'''
import glob
import sys
from os.path import join
import os

import torch
import torch.nn as nn
import numpy as np
from skimage import color
from PIL import Image
from skimage import color
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torchvision.transforms as T
from tqdm import tqdm
from scipy.ndimage.filters import uniform_filter, gaussian_filter

import config as c
if len(sys.argv) > 2:
    c.val_start = int(sys.argv[2])
    c.val_stop = int(sys.argv[3])

if c.no_cond_net:
    import model_no_cond as model
else:
    import model

import data
from data import test_loader

# Some global definitions:
# =========================
# Whether to use the joint bilateral filter for upsampling (slow but better quality)
JBF_FILTER = True
# Use only a selection of val images, e.g.
# VAL_SELECTION = [0,1,5,15]
# per default uses all:
VAL_SELECTION = list(range(len(data.test_list)))

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = c.filename

model.load(model_name)

model.combined_model.eval()
model.combined_model.module.inn.eval()

if not c.no_cond_net:
    model.combined_model.module.feature_network.eval()
    model.combined_model.module.fc_cond_network.eval()

def show_imgs(imgs, save_as):
    '''Save a set of images in a directory (e.g. a set of diverse colorizations
    for a single grayscale image)
    imgs:       List of 3xWxH images (numpy or torch tensors), or Nx3xWxH torch tensor
    save_as:    directory name to save the images in'''


    imgs_np = []

    for im in imgs:
        try:
            im_np = im.data.cpu().numpy()
            imgs_np.append(im_np)
        except:
            imgs_np.append(im)

    try:
        os.mkdir(join(c.img_folder, save_as))
    except OSError:
        pass

    for i, im in enumerate(imgs_np):
        im = np.transpose(im, (1,2,0))

        if im.shape[2] == 1:
            im = np.concatenate([im]*3, axis=2)

        plt.imsave(join(c.img_folder, save_as, '%.2i' % (i)), im)

# Run a single batch to infer the shapes etc.:

for x in test_loader:
    test_set = x
    break

with torch.no_grad():
    x_l, x_ab, cond, ab_pred = model.prepare_batch(test_set)

    outputs = model.cinn(x_ab, cond)
    jac = model.cinn.jacobian(run_forward=False)
    tot_output_size = 2 * c.img_dims[0] * c.img_dims[1]

def sample_z(N, T=1.0):
    ''' Sample N latent vectors, with a sampling temperature T'''
    sampled_z = []
    for o in outputs:
        shape = list(o.shape)
        shape[0] = N
        sampled_z.append(torch.randn(shape).cuda())

    return sampled_z

def sample_resolution_levels(level, z_fixed, N=8, temp=1.):
    '''Generate images with latent code `z_fixed`, but replace the latent dimensions
    at resolution level `level` with random ones.
    N:      number of random samples
    temp:   sampling temperature
    naming of output files: <sample index>_<level>_<val. index>.png'''

    assert len(test_loader) == 1, "please use only one batch worth of images"

    for n in range(N):
        counter = 0
        for x in tqdm(test_loader):
            with torch.no_grad():

                z = sample_z(x.shape[0], temp)
                z_fixed[3-level] = z[3-level]

                x_l, x_ab, cond, ab_pred = model.prepare_batch(x)

                ab_gen = model.combined_model.module.reverse_sample(z_fixed, cond)
                rgb_gen = data.norm_lab_to_rgb(x_l.cpu(), ab_gen.cpu(), filt=True)

            for im in rgb_gen:
                im = np.transpose(im, (1,2,0))
                plt.imsave(join(c.img_folder, '%.6i_%i_%.3i.png' % (counter, level, n)), im)
                counter += 1

def colorize_batches(temp=1., postfix=0, filt=True):
    '''Colorize the whole validation set once.
    temp:       Sampling temperature
    postfix:    Has to be int. Append to file name (e.g. make 10 diverse colorizations of val. set)
    filt:       Whether to use JBF
    '''
    counter = 0
    for x in tqdm(test_loader):
        with torch.no_grad():
            z = sample_z(x.shape[0], temp)
            x_l, x_ab, cond, ab_pred = model.prepare_batch(x)

            ab_gen = model.combined_model.module.reverse_sample(z, cond)
            rgb_gen = data.norm_lab_to_rgb(x_l.cpu(), ab_gen.cpu(), filt=filt)

        for im in rgb_gen:
            im = np.transpose(im, (1,2,0))
            plt.imsave(join(c.img_folder, '%.6i_%.3i.png' % (counter, postfix)), im)
            counter += 1

def interpolation_grid(val_ind=0, grid_size=5, max_temp=0.9, interp_power=2):
    '''
    Make a grid of a 2D latent space interpolation.
    val_ind:        Which image to use (index in current val. set)
    grid_size:      Grid size in each direction
    max_temp:       Maximum temperature to scale to in each direction (note that the corners
                    will have temperature sqrt(2)*max_temp
    interp_power:   Interpolate with (linspace(-lim**p, +lim**p))**(1/p) instead of linear.
                    Because little happens between t = 0.0...0.7, we don't want this to take up the
                    whole grid. p>1 gives more space to the temperatures closer to 1.
    '''
    steps = np.linspace(-(max_temp**interp_power), max_temp**interp_power, grid_size, endpoint=True)
    steps = np.sign(steps) * np.abs(steps)**(1./interp_power)

    test_im = []
    for i,x in enumerate(test_loader):
      test_im.append(x)

    test_im = torch.cat(test_im, dim=0)
    test_im = torch.stack([test_im[i] for i in VAL_SELECTION], dim=0)
    test_im = torch.cat([test_im[val_ind:val_ind+1]]*grid_size**2, dim=0).cuda()


    def interp_z(z0, z1, a0, a1):
        z_out = []
        for z0_i, z1_i in zip(z0, z1):
            z_out.append(a0 * z0_i + a1 * z1_i)
        return z_out

    torch.manual_seed(c.seed+val_ind)
    z0 = sample_z(1, 1.)
    z1 = sample_z(1, 1.)

    z_grid = []
    for dk in steps:
        for dl in steps:
            z_grid.append(interp_z(z0, z1, dk, dl))

    z_grid = [torch.cat(z_i, dim=0) for z_i in list(map(list, zip(*z_grid)))]

    with torch.no_grad():
        x_l, x_ab, cond, ab_pred = model.prepare_batch(test_im)
        ab_gen = model.combined_model.module.reverse_sample(z_grid, cond)

    rgb_gen = data.norm_lab_to_rgb(x_l.cpu(), ab_gen.cpu(), filt=True)

    for i,im in enumerate(rgb_gen):
        im = np.transpose(im, (1,2,0))
        plt.imsave(join(c.img_folder, '%.6i_%.3i.png' % (val_ind, i)), im)

def flow_visualization(val_ind=0, n_samples=2):

    test_im = []
    for i,x in enumerate(test_loader):
      test_im.append(x)

    test_im = torch.cat(test_im, dim=0)
    test_im = torch.stack([test_im[i] for i in VAL_SELECTION], dim=0)
    test_im = torch.cat([test_im[val_ind:val_ind+1]]*n_samples, dim=0).cuda()

    torch.manual_seed(c.seed)
    z = sample_z(n_samples, 1.)

    block_idxs = [(1,7), (11,13), (14,18), (19,24), (28,32),
                  (34,44), (48,52), (54,64), (68,90)]
    block_steps = [12, 10, 10, 10, 12, 12, 10, 16, 12]

    #scales = [0.9, 0.9, 0.7, 0.5, 0.5, 0.2]
    z_levels = [3,5,7]
    min_max_final = None

    def rescale_min_max(ab, new_min, new_max, soft_factor=0.):
        min_ab = torch.min(torch.min(ab, 3, keepdim=True)[0], 2, keepdim=True)[0]
        max_ab = torch.max(torch.max(ab, 3, keepdim=True)[0], 2, keepdim=True)[0]

        new_min = (1. - soft_factor) * new_min - soft_factor * 6
        new_max = (1. - soft_factor) * new_max + soft_factor * 6

        ab = (ab - min_ab) / (max_ab - min_ab)
        return ab * (new_max - new_min) + new_min

    with torch.no_grad():
        x_l, x_ab, cond, ab_pred = model.prepare_batch(test_im)
        x_l_flat = torch.zeros(x_l.shape)
        #x_l_flat *= x_l.mean().item()

        frame_counter = 0

        for level, (k_start, k_stop) in enumerate(block_idxs):
            print('level', level)
            interp_steps = block_steps[level]
            scales = np.linspace(1., 1e-3, interp_steps + 1)
            scales = scales[1:] / scales[:-1]

            for i_interp in tqdm(range(interp_steps)):

                ab_gen = model.combined_model.module.reverse_sample(z, cond).cpu()
                ab_gen = torch.Tensor([[gaussian_filter(x, sigma=2. * (frame_counter / sum(block_steps))) for x in ab] for ab in ab_gen])

                if min_max_final is None:
                    min_max_final = (torch.min(torch.min(ab_gen, 3, keepdim=True)[0], 2, keepdim=True)[0],
                                     torch.max(torch.max(ab_gen, 3, keepdim=True)[0], 2, keepdim=True)[0])
                else:
                    ab_gen = rescale_min_max(ab_gen, *min_max_final,
                                             soft_factor=(frame_counter/sum(block_steps))**2)

                if frame_counter == 0:
                    rgb_gen = data.norm_lab_to_rgb(x_l.cpu(), ab_gen, filt=True)
                    for j in range(rgb_gen.shape[0]):
                        im = rgb_gen[j]
                        im = np.transpose(im, (1,2,0))
                        plt.imsave(join(c.img_folder, 'flow/%.6i_%.3i_final_merged.png' % (val_ind, j+12)), im)

                colors_gen = data.norm_lab_to_rgb(x_l_flat, (1. + 0.2 * (frame_counter / sum(block_steps))) * ab_gen, filt=False)

                for j,im in enumerate(colors_gen):
                    im = np.transpose(im, (1,2,0))
                    im_color =  np.transpose(colors_gen[j], (1,2,0))
                    #plt.imsave(join(c.img_folder, 'flow/%.6i_%.3i_%.3i.png' % (val_ind, j, frame_counter)), im)
                    plt.imsave(join(c.img_folder, 'flow/%.6i_%.3i_%.3i_c.png' % (val_ind, j+12, frame_counter)), im_color)
                frame_counter += 1

                #if level in z_levels:
                    #z[z_levels.index(level)] *= scales[i_interp]
                    #z[-1] *= 1.1

                for k_block in range(k_start,k_stop+1):
                    for key,p in model.combined_model.module.inn.named_parameters():
                        split = key.split('.')
                        if f'module_list.{k_block}.' in key and p.requires_grad:
                            split = key.split('.')
                            if len(split) > 3 and split[3][-1] == '3' and split[2] != 'subnet':
                                p.data *= scales[i_interp]

            for k in range(k_start,k_stop+1):
                for k,p in model.combined_model.module.inn.named_parameters():
                    if f'module_list.{i}.' in k and p.requires_grad:
                        p.data *= 0.0

            #if level in z_levels:
                #z[z_levels.index(level)] *= 0

    state_dict = torch.load(model_name)['net']
    orig_state = model.combined_model.state_dict()
    for name, param in state_dict.items():
        if 'tmp_var' in name:
            continue
        if isinstance(param, nn.Parameter):
            param = param.data
        try:
            orig_state[name].copy_(param)
        except RuntimeError:
            print()
            print(name)
            print()
            raise


def colorize_test_set():
  '''This function is deprecated, for the sake of `colorize_batches`.
  It loops over the image index at the outer level and diverse samples at inner level,
  so it may be useful if you want to adapt it.'''
  test_set = []
  for i,x in enumerate(test_loader):
      test_set.append(x)

  test_set = torch.cat(test_set, dim=0)
  test_set = torch.stack([test_set[i] for i in VAL_SELECTION], dim=0)

  with torch.no_grad():
    temperatures = []

    rgb_bw = data.norm_lab_to_rgb(x_l.cpu(), 0.*x_ab.cpu(), filt=False)
    rgb_gt = data.norm_lab_to_rgb(x_l.cpu(), x_ab.cpu(), filt=JBF_FILTER)

    for i, o in enumerate(outputs):
        std = torch.std(o).item()
        temperatures.append(1.0)

    zz = sum(torch.sum(o**2, dim=1) for o in outputs)
    log_likeli = 0.5 * zz - jac
    log_likeli /= tot_output_size
    print()
    print(torch.mean(log_likeli).item())
    print()

    def sample_z(N, temps=temperatures):
        sampled_z = []
        for o, t in zip(outputs, temps):
            shape = list(o.shape)
            shape[0] = N
            sampled_z.append(t * torch.randn(shape).cuda())

        return sampled_z

    N = 9
    sample_new = True

    for i,n in enumerate(VAL_SELECTION):
        print(i)
        x_i = torch.cat([test_set[i:i+1]]*N, dim=0)
        x_l_i, x_ab_i, cond_i, ab_pred_i = model.prepare_batch(x_i)
        if sample_new:
            z = sample_z(N)

        ab_gen = model.combined_model.module.reverse_sample(z, cond_i)
        rgb_gen = data.norm_lab_to_rgb(x_l_i.cpu(), ab_gen.cpu(), filt=JBF_FILTER)

        i_save = n
        if c.val_start:
            i_save += c.val_start
        show_imgs([rgb_gt[i], rgb_bw[i]] + list(rgb_gen), '%.6i_%.3i' % (i_save, i))

def color_transfer():
  '''Transfers latent code from images to some new conditioning image (see paper Fig. 13)
  Uses images from the directory ./transfer. See code for changing which images are used.'''

  with torch.no_grad():
    cond_images = []
    ref_images = []
    images = ['00', '01', '02']
    for im in images:
        cond_images += [F'./transfer/{im}_c.jpg']*3
        ref_images += [F'./transfer/{im}_{j}.jpg' for j in range(3)]

    def load_image(fname):
        im = Image.open(fname)
        im = data.transf_test(im)
        im = data.test_data.to_tensor(im).numpy()
        im = np.transpose(im, (1,2,0))
        im = color.rgb2lab(im).transpose((2, 0, 1))

        for i in range(3):
            im[i] = (im[i] - data.offsets[i]) / data.scales[i]
        return torch.Tensor(im)

    cond_inputs = torch.stack([load_image(f) for f in cond_images], dim=0)
    ref_inputs = torch.stack([load_image(f) for f in ref_images], dim=0)

    L, x, cond, _ = model.prepare_batch(ref_inputs)
    L_new, _, cond_new, _ = model.prepare_batch(cond_inputs)

    z = model.combined_model.module.inn(x, cond)
    z_rand = sample_z(len(ref_images))

    for zi in z:
        print(zi.shape)

    for i, (s,t) in enumerate([(1.0,1), (0.7,1), (0.0,1.0), (0,1.0)]):
        z_rand[i] = np.sqrt(s) * z_rand[i] + np.sqrt(1.-s) * z[i]

    x_new = model.combined_model.module.reverse_sample(z_rand, cond_new)

    im_ref = data.norm_lab_to_rgb(L.cpu(), x.cpu(), filt=True)
    im_cond = data.norm_lab_to_rgb(L_new.cpu(), 0*x_new.cpu(), bw=True)
    im_new = data.norm_lab_to_rgb(L_new.cpu(), x_new.cpu(), filt=True)

    for i, im in enumerate(ref_images):
        show_imgs([im_ref[i], im_cond[i], im_new[i]], im.split('/')[-1].split('.')[0])

def find_map():
    '''For a given conditioning, try to find the maximum likelihood colorization.
    It doesn't work, but I left in the function to play around with'''

    import torch.nn as nn
    import torch.optim
    z_optim = []
    parameters = []

    z_random = sample_z(4*len(VAL_SELECTION))
    for i, opt in enumerate([False]*2 + [True]*2):
        if opt:
            z_optim.append(nn.Parameter(z_random[i]))
            parameters.append(z_optim[-1])
        else:
            z_optim.append(z_random[i])

    optimizer = torch.optim.Adam(parameters, lr = 0.1)#, momentum=0.0, weight_decay=0)

    cond_4 = [torch.cat([c]*4, dim=0) for c in cond]
    for i in range(100):
        for k in range(10):
            optimizer.zero_grad()
            zz = sum(torch.sum(o**2, dim=1) for o in z_optim)
            x_new = model.combined_model.module.reverse_sample(z_optim, cond_4)
            jac = model.combined_model.module.inn.jacobian(run_forward=False, rev=True)

            log_likeli = 0.5 * zz + jac
            log_likeli /= tot_output_size

            log_likeli = (torch.mean(log_likeli)
                          # Regularizer: variance within image
                          + 0.1 * torch.mean(torch.log(torch.std(x_new[:, 0].view(4*len(VAL_SELECTION), -1), dim=1))**2
                                           + torch.log(torch.std(x_new[:, 1].view(4*len(VAL_SELECTION), -1), dim=1))**2)
                          # Regularizer: variance across images
                          + 0.1 * torch.mean(torch.log(torch.std(x_new, dim=0))**2))

            log_likeli.backward()
            optimizer.step()

        if (i%10) == 0:
            show_imgs(list(data.norm_lab_to_rgb(torch.cat([x_l]*4, 0), x_new, filt=False)), '%.4i' % i)

        print(i, '\t', log_likeli.item(), '\t', 0.25 * sum(torch.std(z_optim[k]).item() for k in range(4)))

def latent_space_pca(img_names = ['zebra']):
    '''This wasn't used in the paper or worked on in a while.
    Perform PCA on latent space to see where images lie in relation to each other.
    See code for details.'''

    image_characteristics = []

    for img_name in img_names:
        img_base = './demo_images/' + img_name
        high_sat = sorted(glob.glob(img_base + '_???.png'))
        #low_sat = sorted(glob.glob(img_base + '_b_???.png'))
        low_sat = []

        to_tensor = T.ToTensor()

        demo_imgs = []
        repr_colors = []

        for fname in high_sat + low_sat:
            print(fname)

            im = plt.imread(fname)
            if img_name == 'zebra':
                repr_colors.append(np.mean(im[0:50, -50:, :], axis=(0,1)))
            elif img_name == 'zebra_blurred':
                repr_colors.append(np.mean(im[0:50, -50:, :], axis=(0,1)))
            elif img_name == 'snowboards':
                repr_colors.append(np.mean(im[50:60, 130:140, :], axis=(0,1)))
            else:
                raise ValueError

            im = color.rgb2lab(im).transpose((2, 0, 1))
            for i in range(3):
                im[i] = (im[i] - data.offsets[i]) / data.scales[i]

            demo_imgs.append(torch.Tensor(im).expand(1, -1, -1, -1))

        demo_imgs = torch.cat(demo_imgs, dim=0)
        x_l, x_ab, cond, ab_pred = model.prepare_batch(demo_imgs)

        outputs = model.cinn(x_ab, cond)
        jac = model.cinn.jacobian(run_forward=False)

        if c.n_downsampling < 2:
            outputs = [outputs]

        outputs_cat = torch.cat(outputs, dim=1)
        outputs_cat = outputs_cat.cpu().numpy()
        jac = jac.cpu().numpy()

        zz = np.sum(outputs_cat**2, axis=1)
        log_likeli = - zz / 2. + np.abs(jac)
        log_likeli /= outputs_cat.shape[1]
        print(log_likeli)
        repr_colors = np.array(repr_colors)

        image_characteristics.append([log_likeli, outputs_cat, repr_colors])


    log_likeli_combined = np.concatenate([C[0] for C in image_characteristics], axis=0)
    outputs_combined = np.concatenate([C[1] for C in image_characteristics], axis=0)

    pca = PCA(n_components=2)
    pca.fit(outputs_combined)

    for i, img_name in enumerate(img_names):
        log_likeli, outputs_cat, repr_colors = image_characteristics[i]


        size = 10 + (40 * (log_likeli - np.min(log_likeli_combined)) / (np.max(log_likeli_combined) - np.min(log_likeli_combined)))**2
        outputs_pca = pca.transform(outputs_cat)
        center = pca.transform(np.zeros((2, outputs_cat.shape[1])))

        plt.figure(figsize=(9,9))
        plt.scatter(outputs_pca[:len(high_sat), 0], outputs_pca[:len(high_sat), 1], s=size[:len(high_sat)], c=repr_colors[:len(high_sat)])
        #plt.scatter(outputs_pca[len(high_sat):, 0], outputs_pca[len(high_sat):, 1], s=size[len(high_sat):], c=repr_colors[len(high_sat):])
        #plt.colorbar()
        #plt.scatter(center[:, 0], center[:, 1], c='black', marker='+', s=150)
        plt.xlim(-100, 100)
        plt.ylim(-100, 100)
        plt.savefig(F'colorspace_{img_name}.png', dpi=200)

if __name__ ==  '__main__':
    pass

    # Comment in which ever you want to run:
    # ========================================

    #for i in tqdm(range(len(data.test_list))):
    for i in [110, 122]:
        print(i)
        flow_visualization(i, n_samples=10)

    #for i in tqdm(range(len(data.test_list))):
        #interpolation_grid(i)

    #latent_space_pca()

    #colorize_test_set()

    #for i in range(8):
        #torch.manual_seed(i+c.seed)
        #colorize_batches(postfix=i, temp=1.0, filt=False)

    #for i  in range(6):
        #torch.manual_seed(c.seed)
        #z_fixed = sample_z(outputs[0].shape[0], 0.0000001)
        #sample_resolution_levels(i, z_fixed)

    #color_transfer()

    #find_map()
