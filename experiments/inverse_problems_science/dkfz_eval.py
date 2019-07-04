import pickle

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch

import dkfz_train
import model
import config as c

model.load('output/dkfz_inn.pt')

print('Trainable parameters:')
print(sum([p.numel() for p in model.params_trainable]))

def concatenate_test_set():
    x_all, y_all = [], []

    for x,y in c.test_loader:
        x_all.append(x)
        y_all.append(y)

    return torch.cat(x_all, 0), torch.cat(y_all, 0)

x_all, y_all = concatenate_test_set()

def sample_posterior(y_it, N=4096):

    outputs = []
    for y in y_it:

        rev_inputs = torch.cat([torch.randn(N, c.ndim_z + c.ndim_pad_zy),
                                torch.zeros(N, c.ndim_y)], 1).to(c.device)

        if c.ndim_pad_zy:
            rev_inputs[:, c.ndim_z:-c.ndim_y] *= c.add_pad_noise 

        rev_inputs[:, -c.ndim_y:] = y

        with torch.no_grad():
            x_samples =  model.model(rev_inputs, rev=True)
        outputs.append(x_samples.data.cpu().numpy())

    return outputs

def show_posteriors():
    # how many different posteriors to show:
    n_plots = 5

    # how many dimensions of x to use:
    n_x = 3

    def hists(x):
        results = []
        for j in range(n_x):
            h, b = np.histogram(x[:, j], bins=100, range=(-2,2), density=True)
            h /= np.max(h)
            results.append([b[:-1],h])
        return results

    prior_hists = hists(x_all)

    x_gt = x_all[:n_plots]
    y_gt = y_all[:n_plots]

    posteriors = sample_posterior(y_gt)

    confidence = 0.68
    q_low  = 100. * 0.5 * (1 - confidence)
    q_high = 100. * 0.5 * (1 + confidence)

    for i in range(n_plots):
        hist_i = hists(posteriors[i])

        for j in range(n_x):
            plt.subplot(n_plots, n_x, n_x*i + j + 1)
            plt.step(*(prior_hists[j]), where='post', color='grey') 
            plt.step(*(hist_i[j]), where='post', color='blue')

            x_low, x_high = np.percentile(posteriors[i][:,j], [q_low, q_high])
            plt.plot([x_gt[i,j], x_gt[i,j]], [0,1], color='black')
            plt.plot([x_low, x_low], [0,1], color='orange')
            plt.plot([x_high, x_high], [0,1], color='orange')

    plt.tight_layout()

def calibration_error():

    # which parameter to look at (0: SO2)
    x_ind = 0
    # how many different confidences to look at
    n_steps = 100
    
    q_values = []
    confidences = np.linspace(0., 1., n_steps+1, endpoint=False)[1:]
    uncert_intervals = [[] for i in range(n_steps)]
    inliers = [[] for i in range(n_steps)]

    for conf in confidences:
        q_low  = 0.5 * (1 - conf)
        q_high = 0.5 * (1 + conf)
        q_values += [q_low, q_high]

    from tqdm import tqdm
    for x,y in tqdm(zip(x_all, y_all), total=x_all.shape[0], disable=False):
        post = sample_posterior([y])[0][:, x_ind]
        x_margins = list(np.quantile(post, q_values))

        for i in range(n_steps):
            x_low, x_high = x_margins.pop(0), x_margins.pop(0) 

            uncert_intervals[i].append(x_high - x_low)
            inliers[i].append(int(x[x_ind] < x_high and x[x_ind] > x_low))


    inliers = np.mean(inliers, axis=1)
    uncert_intervals = np.median(uncert_intervals, axis=1)
    calib_err = inliers - confidences

    print(F'Median calibration error:               {np.median(np.abs(calib_err))}')
    print(F'Calibration error at 68% confidence:    {calib_err[68]}')
    print(F'Med. est. uncertainty at 68% conf.:     {uncert_intervals[68]}')

    plt.subplot(2, 1, 1)
    plt.plot(confidences, calib_err)
    plt.ylabel('Calibration error')

    plt.subplot(2, 1, 2)
    plt.plot(confidences, uncert_intervals)
    plt.ylabel('Median estimated uncertainty')
    plt.xlabel('Confidence')

show_posteriors()
calibration_error()
plt.show()


