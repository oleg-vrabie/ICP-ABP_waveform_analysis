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
import matplotlib
# =============================================================================
pwd = '/home/ov/data/stacked/selected/trial2/'
scids = ['scp', 'nsc']
wfs = [1, 2, 3]
# =============================================================================
# Loading waves by scid in waves and by wf in wf_waves
waves = np.zeros((0))
xs_list = []
ys_list = []
zs_list = []

fitted_list = []

for scid in scids:
    wf_waves = np.zeros((0))
    for wf in wfs:
        # For full PCA + clustering
        waves = utils.load_waves(pwd + 'wf{}_{}.npy'.format(wf, scid), waves)
        print('Loaded: wf{}_{}.npy'.format(wf, scid))

        # For stack_plot
        wf_waves = utils.load_waves(pwd + 'wf{}_{}.npy'.format(wf, scid), wf_waves)

    fitted = utils.pca_projections(wf_waves, 3, svd_solver='arpack')
    fitted_list.append(fitted)
    xs_list.append(fitted[:, 0])
    ys_list.append(fitted[:, 1])
    zs_list.append(fitted[:, 2])
# =============================================================================
# Plot PCA proj. of ICP corresponding to each wf:
title = 'Projection of ICP waves corresponding to all wfs'
utils.stack_plot(xs_list, ys_list, zs_list,
                 label_prefix = 'scid',
                 title = title)
# =============================================================================
# Fit PCA and plot
fitted = utils.pca_projections(waves, 3, svd_solver='arpack')
print('Fitted PCA: {}'.format(fitted.shape))
xs = fitted[:, 0]
ys = fitted[:, 1]
zs = fitted[:, 2]
figu, axu = utils.plot_3d_mod(xs, ys, zs,
                    title='all icp waves from selected wfs',
                    s=10,
                    edge = 'navy',
                    xlabel = 'PC1',
                    ylabel = 'PC2',
                    zlabel = 'PC3',
                    add_info = '({})'.format(scid))

if input('\nSatisfied? (y|N)\n') != 'y':
    raise Exception('Not satisified :(')

# =============================================================================
# Gaussian Mixture Models OR HDBSCAN and, if needed, additional clustering *
decision = input('dbscan or gmm: ')
if decision == 'gmm':
    n_components = int(input('Type the number of components to split into: '))

    cluster_means, clusters, ns, ks, clusters_list = utils.gaussian_mixture_pca_projections(
                                                    fitted, waves, n_components,
                                                    npat=None, scid=scid,
                                                    wave_type='icp',
                                                    n_init=1)
# ______________________________________________________________________________
elif decision == 'dbscan':
    labels_list = []
    sub_colors_list = []
    n_clusters_list = []
    xs_list = []
    ys_list = []
    zs_list = []

    for i in range(len(scids)):
        fitted = fitted_list[i]
        print('Clustering {} waves:'.format(scids[i]))
        labels, sub_colors, n_clusters = utils.hdbscan_(fitted)

        labels_list.append(labels)
        sub_colors_list.append(sub_colors)
        n_clusters_list.append(n_clusters)

        xs_list.append(fitted[sub_colors[:] != -1, 0])
        ys_list.append(fitted[sub_colors[:] != -1, 1])
        zs_list.append(fitted[sub_colors[:] != -1, 2])

    # Plot PCA proj. of ICP:
    title = 'Projection of ICP waves corresponding to all wfs'
    utils.stack_plot(xs_list, ys_list, zs_list,
                     label_prefix = 'scid',
                     title = title)
# ______________________________________________________________________________
else:
    print('EoF')
