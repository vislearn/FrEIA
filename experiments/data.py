import numpy as np
import torch
import torch.utils.data

verts = [
         (-2.4142, 1.),
         (-1., 2.4142),
         (1.,  2.4142),
         (2.4142,  1.),
         (2.4142, -1.),
         (1., -2.4142),
         (-1., -2.4142),
         (-2.4142, -1.)
        ]

label_maps = {
              'all':  [0, 1, 2, 3, 4, 5, 6, 7],
              'some': [0, 0, 0, 0, 1, 1, 2, 3],
              'none': [0, 0, 0, 0, 0, 0, 0, 0],
             }


def generate(labels, tot_dataset_size):
    # print('Generating artifical data for setup "%s"' % (labels))

    np.random.seed(0)
    N = tot_dataset_size
    mapping = label_maps[labels]

    pos = np.random.normal(size=(N, 2), scale=0.2)
    labels = np.zeros((N, 8))
    n = N//8

    for i, v in enumerate(verts):
        pos[i*n:(i+1)*n, :] += v
        labels[i*n:(i+1)*n, mapping[i]] = 1.

    shuffling = np.random.permutation(N)
    pos = torch.tensor(pos[shuffling], dtype=torch.float)
    labels = torch.tensor(labels[shuffling], dtype=torch.float)

    return pos, labels

    # test_loader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(pos[:test_split], labels[:test_split]),
    #     batch_size=batch_size, shuffle=True, drop_last=True)

    # train_loader = torch.utils.data.DataLoader(
    #     torch.utils.data.TensorDataset(pos[test_split:], labels[test_split:]),
    #     batch_size=batch_size, shuffle=True, drop_last=True)

    # return test_loader, train_loader
