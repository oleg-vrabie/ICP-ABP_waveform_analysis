from sklearn.decomposition import PCA as PCA
import scipy.io
import numpy as np
import inspect

import sys
sys.path.append('/home/ov/python/py_utils')
import utils

patient_nr = 10
scid = 'nsc'

pwd = '/home/ov/preprocessed_waves/extracted/'
pwd_means = pwd + 'nfpat' + str(patient_nr) + '.icp.' + scid + '_means'
cluster_means = np.loadtxt(pwd_means)

print(cluster_means.shape)

# Normalization:
from sklearn.preprocessing import normalize
cluster_means = normalize(cluster_means, norm='max')

fig = utils.plot_means_a4(cluster_means, title='PAT {} {} (pa)'.format(patient_nr,
                                                                 scid))  # pa := postapocalyptic
