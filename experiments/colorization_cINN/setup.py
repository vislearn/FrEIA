#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("joint_bilateral_filter",
                             sources=["joint_bilateral_filter.pyx", "joint_bilateral.c"],
                             include_dirs=[np.get_include()])],
)

import joint_bilateral_filter as jbf
import matplotlib.pyplot as plt
from skimage.color import lab2rgb
from skimage.transform import resize

L = plt.imread('./demo_images/L.png')[:, :, 0]

ab = np.stack([plt.imread('./demo_images/a.png')[::2, ::2, 0],
               plt.imread('./demo_images/b.png')[::2, ::2, 0]], axis=0)

s_x = 4.
s_l = 0.05

print('===='*20 + '\n\n' + '===='*20)
ab_up = jbf.upsample(L, ab, s_x, s_l)
ab_naive0 = resize(ab.transpose((1,2,0)), (ab_up.shape[1], ab_up.shape[2]), order=0).transpose((2,0,1))
ab_naive1 = resize(ab.transpose((1,2,0)), (ab_up.shape[1], ab_up.shape[2]), order=1).transpose((2,0,1))

Lab_up = np.concatenate([L[np.newaxis, :, :], ab_up], axis=0).transpose((1,2,0))
Lab_naive0 = np.concatenate([L[np.newaxis, :, :], ab_naive0], axis=0).transpose((1,2,0))
Lab_naive1 = np.concatenate([L[np.newaxis, :, :], ab_naive1], axis=0).transpose((1,2,0))

for i, s, o in [(0, 100, 0),
                (1, 255, -128),
                (2, 255, -128)]:

    Lab_up[:, :, i] = Lab_up[:, :, i] * s + o
    Lab_naive0[:, :, i] = Lab_naive0[:, :, i] * s + o
    Lab_naive1[:, :, i] = Lab_naive1[:, :, i] * s + o

rgb_up = lab2rgb(Lab_up)
rgb_naive0 = lab2rgb(Lab_naive0)
rgb_naive1 = lab2rgb(Lab_naive1)

plt.figure()
plt.title('nearest')
plt.imshow(rgb_naive0)
plt.figure()
plt.title('bilinear')
plt.imshow(rgb_naive1)
plt.figure()
plt.title('joint bilateral')
plt.imshow(rgb_up)

plt.show()
