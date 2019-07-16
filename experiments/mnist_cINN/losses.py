import torch
import numpy as np
from torch.autograd import Variable

import config as c

def MMD(x, y):
    xx, yy, xy = torch.mm(x,x.t()), torch.mm(y,y.t()), torch.mm(x,y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.*xx
    dyy = ry.t() + ry - 2.*yy
    dxy = rx.t() + ry - 2.*xy

    dxx = torch.clamp(dxx, 0., np.inf)
    dyy = torch.clamp(dyy, 0., np.inf)
    dxy = torch.clamp(dxy, 0., np.inf)

    XX, YY, XY = (Variable(torch.zeros(xx.shape).cuda()),
                  Variable(torch.zeros(xx.shape).cuda()),
                  Variable(torch.zeros(xx.shape).cuda()))

    for cw in c.kernel_widths:
        for a in c.kernel_powers:
            XX += cw**a * (cw + 0.5 * dxx / a)**-a
            YY += cw**a * (cw + 0.5 * dyy / a)**-a
            XY += cw**a * (cw + 0.5 * dxy / a)**-a

    return torch.mean(XX + YY - 2.*XY)

def moment_match(x, y):
    return (torch.mean(x) - torch.mean(y))**2 + (torch.var(x) - torch.var(y))**2
