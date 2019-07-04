import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

import config as c

def MMD_matrix_multiscale(x, y, widths_exponents):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = torch.clamp(rx.t() + rx - 2.*xx, 0, np.inf)
    dyy = torch.clamp(ry.t() + ry - 2.*yy, 0, np.inf)
    dxy = torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

    XX, YY, XY = (torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device),
                  torch.zeros(xx.shape).to(c.device))

    for C,a in widths_exponents:
        XX += C**a * ((C + dxx) / a)**-a
        YY += C**a * ((C + dyy) / a)**-a
        XY += C**a * ((C + dxy) / a)**-a

    return XX + YY - 2.*XY

def l2_dist_matrix(x, y):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    return torch.clamp(rx.t() + ry - 2.*xy, 0, np.inf)

def forward_mmd(y0, y1):
    return MMD_matrix_multiscale(y0, y1, c.mmd_forw_kernels)

def backward_mmd(x0, x1):
    return MMD_matrix_multiscale(x0, x1, c.mmd_back_kernels)

def l2_fit(input, target):
    return torch.sum((input - target)**2) / c.batch_size

def debug_mmd_terms(XX, YY, XY):

    plt.figure()

    plt.subplot(2,2,1)
    plt.imshow((XX + YY - XY - XY.t()).data.numpy(), cmap='jet')
    plt.title('Tot')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.imshow(XX.data.numpy(), cmap='jet')
    plt.title('XX')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.imshow(YY.data.numpy(), cmap='jet')
    plt.title('YY')
    plt.colorbar()

    plt.subplot(2,2,4)
    plt.imshow(XY.data.numpy(), cmap='jet')
    plt.title('XY')
    plt.colorbar()

    plt.show()
