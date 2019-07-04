from time import time

from tqdm import tqdm
import torch
import torch.optim
import numpy as np

import model
import data

cinn = model.ColorizationCINN(1e-3)
cinn.cuda()
scheduler = torch.optim.lr_scheduler.StepLR(cinn.optimizer, 1, gamma=0.1)

N_epochs = 3
t_start = time()
nll_mean = []

print('Epoch\tBatch/Total \tTime \tNLL train\tNLL val\tLR')
for epoch in range(N_epochs):
    for i, Lab in enumerate(data.train_loader):
        Lab = Lab.cuda()
        z, log_j = cinn(Lab)

        nll = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total
        nll.backward()
        nll_mean.append(nll.item())
        cinn.optimizer.step()
        cinn.optimizer.zero_grad()

        if not i % 20:
            with torch.no_grad():
                z, log_j = cinn(data.val_all[:512])
                nll_val = torch.mean(z**2) / 2 - torch.mean(log_j) / model.ndim_total

            print('%.3i \t%.5i/%.5i \t%.2f \t%.6f\t%.6f\t%.2e' % (epoch,
                                                            i, len(data.train_loader),
                                                            (time() - t_start)/60.,
                                                            np.mean(nll_mean),
                                                            nll_val.item(),
                                                            cinn.optimizer.param_groups[0]['lr'],
                                                            ), flush=True)
            nll_mean = []

    scheduler.step()
torch.save(cinn.state_dict(), f'output/lsun_cinn.pt')
