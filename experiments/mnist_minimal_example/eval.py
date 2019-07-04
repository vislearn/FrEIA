import torch
import numpy as np
import matplotlib.pyplot as plt

import model
import data

cinn = model.MNIST_cINN(0)
cinn.cuda()
state_dict = {k:v for k,v in torch.load('output/mnist_cinn.pt').items() if 'tmp_var' not in k}
cinn.load_state_dict(state_dict)

cinn.eval()

def show_samples(label):
    '''produces and shows cINN samples for a given label (0-9)'''

    N_samples = 100
    l = torch.cuda.LongTensor(N_samples)
    l[:] = label

    z = 1.0 * torch.randn(N_samples, model.ndim_total).cuda()

    with torch.no_grad():
        samples = cinn.reverse_sample(z, l).cpu().numpy()
        samples = data.unnormalize(samples)

    full_image = np.zeros((28*10, 28*10))

    for k in range(N_samples):
        i, j = k // 10, k % 10
        full_image[28 * i : 28 * (i + 1),
                   28 * j : 28 * (j + 1)] = samples[k, 0]

    full_image = np.clip(full_image, 0, 1)
    plt.figure()
    plt.title(F'Generated digits for c={label}')
    plt.imshow(full_image, vmin=0, vmax=1, cmap='gray')

def val_loss():
    '''prints the final validiation loss of the model'''

    with torch.no_grad():
        z, log_j = cinn(data.val_x, data.val_l)
        nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total

    print('Validation loss:')
    print(nll_val.item())

val_loss()

for i in range(10):
    show_samples(i)

plt.show()
