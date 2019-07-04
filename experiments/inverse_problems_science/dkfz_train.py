import numpy as np
import torch
import torch.utils.data

import config as c

x_test = torch.Tensor(np.load('./dkfz_data/x_test.npy'))
y_test = torch.Tensor(np.load('./dkfz_data/y_test.npy'))

x_train = np.concatenate([np.load('./dkfz_data/x_train.npy'),
                          np.load('./dkfz_data/x_additional.npy')], axis=0)

y_train = np.concatenate([np.load('./dkfz_data/y_train.npy'),
                          np.load('./dkfz_data/y_additional.npy')], axis=0)

# There is a single NaN in the dataset
nan_index = np.unique(np.argwhere(y_train != y_train)[:,0])
x_train = np.delete(x_train, nan_index, axis=0)
y_train = np.delete(y_train, nan_index, axis=0)

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)

c.test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=c.batch_size, shuffle=False, drop_last=True)

c.train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, drop_last=True)

c.ndim_x      = 13
c.ndim_pad_x  = 0

c.ndim_y      = 8
c.ndim_z      = 5
c.ndim_pad_zy = 0

c.mmd_back_weighted = True
c.filename_out = 'output/dkfz_inn.pt'

if __name__ == "__main__":
    import train
    train.main()
