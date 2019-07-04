import numpy as np
import torch
import torch.utils.data

import config as c

verts = [
         (-2.4142,  1.),
         (-1.,  2.4142),
         ( 1.,  2.4142),
         ( 2.4142,  1.),
         ( 2.4142, -1.),
         ( 1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
        ]

label_maps = {
              'all':  [0,1,2,3,4,5,6,7],
              'some': [0,0,0,0,1,1,2,3],
              'none': [0,0,0,0,0,0,0,0],
             }


def make_loaders(setup_type, batch_size):
    np.random.seed(0)

    N = int(1e6)
    test_split = 10000
    mapping = label_maps[setup_type] 

    pos = np.random.normal(size=(N, 2), scale = 0.2)
    labels = np.zeros((N, 8))
    n = N//8

    for i, v in enumerate(verts):
        pos[i*n:(i+1)*n,:] += v
        labels[i*n:(i+1)*n,mapping[i]] = 1.

    shuffling = np.random.permutation(N)
    pos = torch.Tensor(pos[shuffling])
    labels = torch.Tensor(labels[shuffling])

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos[:test_split], labels[:test_split]),
        batch_size=batch_size, shuffle=True, drop_last=True)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(pos[test_split:], labels[test_split:]),
        batch_size=batch_size, shuffle=True, drop_last=True)

    return test_loader, train_loader

from visdom import Visdom
viz = Visdom()
scatter_plot = 999

def show_live_posteriors(out_x, out_y, x, y):
    colors = torch.mm(torch.round(y[:, c.ndim_z:].cpu()), torch.Tensor([np.arange(8)+1]).t()).numpy().astype(int).flatten()
    viz.scatter(X=out_x[:, :2].cpu().data.numpy(),
                Y=colors, win=scatter_plot, opts={'markersize':9})

c.test_loader, c.train_loader = make_loaders('some', c.batch_size)
c.test_time_functions = [show_live_posteriors]

c.ndim_x      = 2
c.ndim_pad_x  = 8

c.ndim_z      = 2
c.ndim_y      = 8
c.ndim_pad_zy = 0

c.lambd_fit_forw       = 1.
c.lambd_mmd_forw       = 50.
c.lambd_reconstruct    = 1.
c.lambd_mmd_back       = 250.

c.init_scale = 0.2
c.hidden_layer_sizes = 64
c.N_blocks = 3

c.filename_out = F'output/toy_modes_test_{c.N_blocks}.pt'

if c.N_blocks == 1:
    c.exponent_clamping = 8.

c.mmd_back_kernels = [(0.1, 0.1), (0.8, 0.5), (0.2, 2)]

if __name__ == "__main__":
    import train

    c.lr_init = 3e-4
    c.final_decay = 0.2
    c.n_epochs = 60
    c.train_backward_mmd = False

    train.main()

    c.filename_in = c.filename_out
    c.filename_out += '_2'
    c.lr_init = 5e-5
    c.final_decay = 0.05
    c.n_epochs = 100
    c.train_backward_mmd = True

    train.main()
