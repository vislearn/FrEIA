from math import exp
import torch
import torch.nn as nn

class subnet_coupling_layer(nn.Module):
    def __init__(self, dims_in, dims_c, F_class, subnet, sub_len, F_args={}, clamp=5.):
        super().__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.conditional = True
        condition_length = sub_len
        self.subnet = subnet

        self.s1 = F_class(self.split_len1 + condition_length, self.split_len2*2, **F_args)
        self.s2 = F_class(self.split_len2 + condition_length, self.split_len1*2, **F_args)

    def e(self, s):
        return torch.exp(self.clamp * 0.636 * torch.atan(s / self.clamp))

    def log_e(self, s):
        return self.clamp * 0.636 * torch.atan(s / self.clamp)

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[0].narrow(1, 0, self.split_len1),
                  x[0].narrow(1, self.split_len1, self.split_len2))
        c_star = self.subnet(torch.cat(c, 1))

        if not rev:
            r2 = self.s2(torch.cat([x2, c_star], 1) if self.conditional else x2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = self.e(s2) * x1 + t2

            r1 = self.s1(torch.cat([y1, c_star], 1) if self.conditional else y1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = self.e(s1) * x2 + t1
            self.last_jac = self.log_e(s1) + self.log_e(s2)

        else: # names of x and y are swapped!
            r1 = self.s1(torch.cat([x1, c_star], 1) if self.conditional else x1)
            s1, t1 = r1[:, :self.split_len2], r1[:, self.split_len2:]
            y2 = (x2 - t1) / self.e(s1)

            r2 = self.s2(torch.cat([y2, c_star], 1) if self.conditional else y2)
            s2, t2 = r2[:, :self.split_len1], r2[:, self.split_len1:]
            y1 = (x1 - t2) / self.e(s2)
            self.last_jac = - self.log_e(s1) - self.log_e(s2)

        return [torch.cat((y1, y2), 1)]

    def jacobian(self, x, c=[], rev=False):
        return torch.sum(self.last_jac, dim=tuple(range(1, self.ndims+1)))

    def output_dims(self, input_dims):
        return input_dims
