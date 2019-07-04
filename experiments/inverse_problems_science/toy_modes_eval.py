import torch
import numpy as np
import matplotlib.pyplot as plt

import toy_modes_train
import config as c

i_blocks = 3
c.N_blocks = i_blocks
if i_blocks == 1:
    c.exponent_clamping = 8.

import model

model.load(F'output/toy_modes_test_{i_blocks}.pt_2')

def print_config():
    config_str = ""
    config_str += "="*80 + "\n"
    config_str += "Config options:\n\n"

    for v in dir(c):
        if v[0]=='_': continue
        s=eval('c.%s'%(v))
        config_str += "  {:25}\t{}\n".format(v,s)

    config_str += "="*80 + "\n"

    print(config_str)

print_config()

def concatenate_test_set():
    x_all, y_all = [], []

    for x,y in c.test_loader:
        x_all.append(x)
        y_all.append(y)

    return torch.cat(x_all, 0), torch.cat(y_all, 0)

x_all, y_all = concatenate_test_set()

def sample_posteriors():

    rev_inputs = torch.cat([torch.randn(y_all.shape[0], c.ndim_z), y_all + 0.01 * torch.randn(y_all.shape)], 1).to(c.device)

    with torch.no_grad():
        x_samples =  model.model(rev_inputs, rev=True)

    x_samples = x_samples.cpu().numpy()
    values = torch.mm(y_all, torch.Tensor([np.arange(8)]).t()).numpy()
    
    plt.figure(figsize=(8,8))
    plt.scatter(x_samples[:,0], x_samples[:,1], c=values.flatten(), cmap='Set1', s=2., vmin=0, vmax=9)
    plt.axis('equal')
    plt.axis([-3,3,-3,3])

sample_posteriors()
plt.tight_layout()
plt.savefig(F'ablation_{i_blocks}.png')
plt.show()

