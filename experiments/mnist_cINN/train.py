#!/usr/bin/env python
import sys

import torch
import torch.nn
import torch.optim
from torch.nn.functional import avg_pool2d, interpolate
from torch.autograd import Variable
import numpy as np
import tqdm

import config as c
import opts
opts.parse(sys.argv)
config_str = ""
config_str += "==="*30 + "\n"
config_str += "Config options:\n\n"

for v in dir(c):
    if v[0]=='_': continue
    s=eval('c.%s'%(v))
    config_str += "  {:25}\t{}\n".format(v,s)

config_str += "==="*30 + "\n"

print(config_str)

import model
import data
import viz
import losses

if c.colorize:
    import cond_net

class dummy_loss(object):
    def item(self):
        return 1.

if c.load_file:
    model.load(c.load_file)

def sample_outputs(sigma):
    return sigma * torch.cuda.FloatTensor(c.batch_size, c.output_dim).normal_()

if c.colorize:
    cond_tensor = torch.zeros(c.batch_size, model.cond_size, *c.img_dims).cuda()

    def make_cond(mask, cond_features):
        cond_tensor[:, 0] = mask[:, 0]
        cond_tensor[:, 1:] = cond_features.view(c.batch_size, -1, 1, 1).expand(-1, -1, *c.img_dims)
        return cond_tensor

else:
    cond_tensor = torch.zeros(c.batch_size, model.cond_size).cuda()
    def make_cond(labels):
        cond_tensor.zero_()
        cond_tensor.scatter_(1, labels.view(-1,1), 1.)
        return cond_tensor

    test_labels = torch.LongTensor((list(range(10))*(c.batch_size//10 + 1))[:c.batch_size]).cuda()
    test_cond = make_cond(test_labels).clone()

try:
    for i_epoch in range(-c.pre_low_lr, c.n_epochs):

        loss_history = []
        data_iter = iter(data.train_loader)

        if i_epoch < 0:
            for param_group in model.optim.param_groups:
                param_group['lr'] = c.lr * 2e-2

        for i_batch, data_tuple in tqdm.tqdm(enumerate(data_iter),
                                              total=min(len(data.train_loader), c.n_its_per_epoch),
                                              leave=False,
                                              mininterval=1.,
                                              disable=(not c.progress_bar),
                                              ncols=83):

            if c.colorize:
                x, labels, masks = data_tuple
                #print()
                #print(x.shape, labels.shape, masks.shape, cond_tensor.shape)
                #torch.Size([512, 3, 28, 28]) torch.Size([512]) torch.Size([512, 1, 28, 28]) torch.Size([512, 65])
                x, labels, masks  = x.cuda(), labels.cuda(), masks.cuda()
                x += c.add_image_noise * torch.cuda.FloatTensor(x.shape).normal_()
                with torch.no_grad():
                    cond_features = cond_net.model.features(masks)
                    cond = make_cond(masks, cond_features)

            else:
                x, labels = data_tuple
                x, labels = x.cuda(), labels.cuda()
                x += c.add_image_noise * torch.cuda.FloatTensor(x.shape).normal_()

                cond = make_cond(labels.cuda())

            output = model.model(x, cond)

            if c.do_fwd:
                zz = torch.sum(output**2, dim=1)
                jac = model.model.log_jacobian(run_forward=False)

                neg_log_likeli = 0.5 * zz - jac

                l = torch.mean(neg_log_likeli)
                l.backward(retain_graph=c.do_rev)
            else:
                l = dummy_loss()

            if c.do_rev:
                samples_noisy = sample_outputs(c.latent_noise) + output.data

                x_rec = model.model(samples_noisy, rev=True)
                l_rev = torch.mean( (x-x_rec)**2 )
                l_rev.backward()
            else:
                l_rev = dummy_loss()

            model.optim_step()
            loss_history.append([l.item(), l_rev.item()])

            if i_batch+1 >= c.n_its_per_epoch:
                # somehow the data loader workers don't shut down automatically
                try:
                    data_iter._shutdown_workers()
                except:
                    pass

                break

        model.weight_scheduler.step()

        epoch_losses = np.mean(np.array(loss_history), axis=0)
        epoch_losses[0] = min(epoch_losses[0], 0)

        if i_epoch > 1 - c.pre_low_lr:
            viz.show_loss(epoch_losses, logscale=False)
            output_orig = output.cpu()
            viz.show_hist(output_orig)

        with torch.no_grad():
            samples = sample_outputs(c.sampling_temperature)

            if not c.colorize:
                cond = test_cond

            rev_imgs = model.model(samples, cond, rev=True)
            ims = [rev_imgs]

        viz.show_imgs(*list(data.unnormalize(i) for i in ims))

        model.model.zero_grad()

        if (i_epoch % c.checkpoint_save_interval) == 0:
            model.save(c.filename + '_checkpoint_%.4i' % (i_epoch * (1-c.checkpoint_save_overwrite)))

    model.save(c.filename)

except:
    if c.checkpoint_on_error:
        model.save(c.filename + '_ABORT')

    raise
