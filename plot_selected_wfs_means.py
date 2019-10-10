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
pwd = '/home/ov/data/stacked/selected/trial1/'
scids = ['scp', 'nsc']
wfs = [1, 2, 3]
# =============================================================================
# Load waves:

# =============================================================================
# Fit-Transforming PCA on acquired waves:
"""
fitted = utils.pca_projections(waves, 3, svd_solver='arpack')
print('Fitted PCA: {}'.format(fitted.shape))
xs = fitted[:, 0]
ys = fitted[:, 1]
zs = fitted[:, 2]

utils.plot_3d_mod(xs, ys, zs,
                    title='wf{} {}'.format(wf, scid),
                    s=1,
                    edge = 'navy',
                    xlabel = 'PC1',
                    ylabel = 'PC2',
                    zlabel = 'PC3',
                    add_info = 'PAT{} ({})'.format(wf, scid))
if input('Satisfied? (y|N)\n') != 'y':
    raise Exception('Not satisified :(')
"""
# =============================================================================
orientation = None

side1 = 2
side2 = 3

top = 0.9
bottom = 0.040
figsize=(8.27,11.69) if orientation=='portrait' else (11.69, 8.27)
fig, ax = plt.subplots(side1, side2, figsize=figsize)
fig.subplots_adjust(left=0.03, right=0.97, top=top, bottom=bottom, hspace=0.220)



for wf in wfs:
    for scid in scids:
        waves = np.load(pwd + 'wf{}_{}.npy'.format(wf, scid))
        print('\nloaded: wf{}_{}.npy'.format(wf, scid))
        mean_wf = np.mean(waves, axis = 0)
        print(mean_wf.shape)
        print('wf{} {}'.format(wfs.index(wf), scids.index(scid)))
        axis = ax[scids.index(scid), wfs.index(wf)]
        axis.plot(np.arange(0, 780, 1), mean_wf, color='black')
        axis.set_title('wf{} {} ({})'.format(wf, scid, waves.shape[0]))

        fitted = utils.pca_projections(waves, 3, svd_solver='arpack')
        print('Fitted PCA: {}'.format(fitted.shape))
        xs = fitted[:, 0]
        ys = fitted[:, 1]
        zs = fitted[:, 2]
        figu, axu = utils.plot_3d_mod(xs, ys, zs,
                            title='wf{} {}'.format(wf, scid),
                            s=1,
                            edge = 'navy',
                            xlabel = 'PC1',
                            ylabel = 'PC2',
                            zlabel = 'PC3',
                            add_info = 'wf{} ({})'.format(wf, scid))
        """
        """
plt.suptitle('Extracted ICP means corresponding to three ABP waveforms (trial1)')
plt.show()
