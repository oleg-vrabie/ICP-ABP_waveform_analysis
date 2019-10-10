# #############################################################################
# Script for studying projections of ABP/ICP waves and the relationship between
# them. It projects any of these wave-types into a lower dimension space using
# PCA and then applies clustering techniques to segregate between waveforms.
# Final step consist of correlating extracted clusters with time-corresponding
# counterparts: ICP - ABP and vice-versa
# #############################################################################
import scipy.io
import numpy as np
import sys
sys.path.append('/home/ov/python/py_utils')
import utils
import matplotlib.pyplot as plt
# =============================================================================
# Parameters
wf = 1
scid = 'nsc'
wave_type = 'abp'
trial = 2
pwd = '/home/ov/data/stacked/selected/trial' + str(trial) + '/'
# =============================================================================
# Load waves:
waves = np.load(pwd + 'wf{}_{}.npy'.format(wf, scid))
plt.plot(waves[1000])
plt.show()
# =============================================================================
# Fit-Transforming PCA on acquired waves:
fitted = utils.pca_projections(waves, 3, svd_solver='arpack')
print('Fitted PCA: {}'.format(fitted.shape))
xs = fitted[:, 0]
ys = fitted[:, 1]
zs = fitted[:, 2]


utils.plot_3d_mod(xs, ys, zs,
                    title='wf{} {}'.format(wf, scid),
                    s=5,
                    edge = 'navy',
                    xlabel = 'PC1',
                    ylabel = 'PC2',
                    zlabel = 'PC3',
                    add_info = 'PAT{} ({})'.format(wf, scid))
if input('Satisfied? (y|N)\n') != 'y':
    raise Exception('Not satisified :(')
# =============================================================================
# Gaussian Mixture Models and, if needed, additional clustering
n_components = int(input('Type the number of components to split into: '))

fitted_copy = fitted

cluster_means, clusters, ns, ks, clusters_list = utils.gaussian_mixture_pca_projections(
                                                fitted, waves, n_components,
                                                npat=None, scid=scid,
                                                wave_type=wave_type,
                                                n_init=5)
