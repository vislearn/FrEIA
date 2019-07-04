#!/usr/bin/env python
import sys

import torch
import torch.nn
import torch.optim
from torch.nn.functional import avg_pool2d#, interpolate
from torch.autograd import Variable
import numpy as np
import tqdm

import config as c

if c.no_cond_net:
    import model_no_cond as model
else:
    import model

import data
import viz

if c.load_file:
    model.load(c.load_file)

class dummy_loss(object):
    def item(self):
        return 1.

def sample_outputs(sigma, out_shape):
    return [sigma * torch.cuda.FloatTensor(torch.Size((4, o))).normal_() for o in out_shape]

tot_output_size = 2 * c.img_dims[0] * c.img_dims[1]

try:
    for i_epoch in range(-c.pre_low_lr, c.n_epochs):

        loss_history = []
        data_iter = iter(data.train_loader)

        if i_epoch < 0:
            for param_group in model.optim.param_groups:
                param_group['lr'] = c.lr * 2e-2
        if i_epoch == 0:
            for param_group in model.optim.param_groups:
                param_group['lr'] = c.lr

        if c.end_to_end and i_epoch <= c.pretrain_epochs:
            for param_group in model.feature_optim.param_groups:
                param_group['lr'] = 0
            if i_epoch == c.pretrain_epochs:
                for param_group in model.feature_optim.param_groups:
                    param_group['lr'] = 1e-4

        iterator = tqdm.tqdm(enumerate(data_iter),
                             total=min(len(data.train_loader), c.n_its_per_epoch),
                             leave=False,
                             mininterval=1.,
                             disable=(not c.progress_bar),
                             ncols=83)

        for i_batch , x in iterator:

            zz, jac = model.combined_model(x)

            neg_log_likeli = 0.5 * zz - jac

            l = torch.mean(neg_log_likeli) / tot_output_size
            l.backward()

            model.optim_step()
            loss_history.append([l.item(), 0.])

            if i_batch+1 >= c.n_its_per_epoch:
                # somehow the data loader workers don't shut down automatically
                try:
                    data_iter._shutdown_workers()
                except:
                    pass

                iterator.close()
                break

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[1] = np.log10(model.optim.param_groups[0]['lr'])
        for i in range(len(epoch_losses)):
            epoch_losses[i] = min(epoch_losses[i], c.loss_display_cutoff)

        with torch.no_grad():
            ims = []
            for x in data.test_loader:
                x_l, x_ab, cond, ab_pred = model.prepare_batch(x[:4])

                for i in range(3):
                    z = sample_outputs(c.sampling_temperature, model.output_dimensions)
                    x_ab_sampled = model.combined_model.module.reverse_sample(z, cond)
                    ims.extend(list(data.norm_lab_to_rgb(x_l, x_ab_sampled)))

                break

        if i_epoch >= c.pretrain_epochs * 2:
            model.weight_scheduler.step(epoch_losses[0])
            model.feature_scheduler.step(epoch_losses[0])

        viz.show_imgs(*ims)
        viz.show_loss(epoch_losses)

        if i_epoch > 0 and (i_epoch % c.checkpoint_save_interval) == 0:
            model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))

    model.save(c.filename)

except:
    if c.checkpoint_on_error:
        model.save(c.filename + '_ABORT')

    raise
finally:
    viz.signal_stop()
