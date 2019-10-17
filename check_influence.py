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
from matplotlib.patches import Patch
# =============================================================================
# Parameters:
scp_mean = 5
nsc_mean = 1
wave_type = 'icp'
scid = 'scp+nsc'
npat = None
pwd = '/home/ov/data/stacked/selected/trial2/'
# =============================================================================
# Load clusters:
clusters_nsc = np.load(pwd + 'nsc_clusters_hdbscan.npy', allow_pickle = True)
clusters_scp = np.load(pwd + 'scp_clusters_hdbscan.npy', allow_pickle = True)
print(clusters_scp[scp_mean].shape)
print(clusters_nsc[nsc_mean].shape)

fitted_scp = utils.pca_projections(clusters_scp[scp_mean], 3, svd_solver='arpack')
fitted_nsc = utils.pca_projections(clusters_nsc[nsc_mean], 3, svd_solver='arpack')

xs_list = []
ys_list = []
zs_list = []

xs_list.append(fitted_scp[:, 0])
xs_list.append(fitted_nsc[:, 0])
ys_list.append(fitted_scp[:, 1])
ys_list.append(fitted_nsc[:, 1])
zs_list.append(fitted_scp[:, 2])
zs_list.append(fitted_nsc[:, 2])

# For scid == 1 == 'scp' and for scid == 2 == 'nsc'
icp_types = np.zeros((fitted_scp.shape[0], )) + 1
icp_types = np.concatenate((icp_types, (np.zeros((fitted_nsc.shape[0], )) + 2)))
# =============================================================================
# Plot PCA proj. of ICP corresponding to each wf:
title = 'Projection of ICP scp&nsc waves from two clusters with same input wfs'
utils.stack_plot(xs_list, ys_list, zs_list,
                label_prefix = 'scid',
                title = title,
                s = 40)
# =============================================================================
# Fit PCA and plot by scid
fitted = np.concatenate((fitted_scp, fitted_nsc), axis=0)
waves = np.concatenate((clusters_scp[scp_mean], clusters_nsc[nsc_mean]), axis=0)
# =============================================================================
# Gaussian Mixture Models OR HDBSCAN and, if needed, additional clustering *
decision = input('dbscan or gmm: ')
if decision == 'gmm':
    n_components = int(input('Type the number of components to split into: '))

    gmm = utils.gaussian_mixture_(fitted, waves, n_components,
                                  npat=None, scid=scid,
                                  wave_type='icp',
                                  n_init=1,
                                  random_state=3)

    predictions = gmm.predict(fitted)

    fig = plt.figure(num='Projection',figsize=(16,9))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.3)
    ax = fig.add_subplot(111, projection='3d')

    legend_elements = []

    for i in range(n_components):
        cluster = np.asarray(fitted[i==predictions])
        centroid = gmm.means_[i]
        xs = np.reshape(cluster, (cluster.shape[0] ,3))[:, 0]
        ys = np.reshape(cluster, (cluster.shape[0] ,3))[:, 1]
        zs = np.reshape(cluster, (cluster.shape[0] ,3))[:, 2]
        print('{} added'.format(i+1))
        print('\t{}/{} of shape {}'.format(i+1, n_components, xs.shape))

        # For the ratios of wfs in the legend
        total_nr = waves[i==predictions].shape[0]/100
        per1 = round(waves[(i==predictions) & (1 == icp_types)].shape[0]/total_nr, 1)
        per2 = round(waves[(i==predictions) & (2 == icp_types)].shape[0]/total_nr, 1)

        legend_elements.append(Patch(facecolor=utils.default_colors[i],
                                     edgecolor='black',
                                     label='{} ({}% scp | {}% nsc)'.format(xs.shape[0],
                                                                     per1,
                                                                     per2)))

        ax.scatter(xs, ys, zs,
                    c=utils.default_colors[i],
                    marker="${}$".format(i+1),
                    depthshade=False,
                    s=60)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(handles=legend_elements, loc='upper right', fontsize = 'large')
    plt.title('GMM of {}({})'
              .format(wave_type, scid))
    plt.show(block=False)
    plt.pause(0.001)

    if input('\nWant the means? (y|N)\n') != 'y':
        print('ok')
    print('That was it.')
# ______________________________________________________________________________
elif decision == 'dbscan':
    labels, sub_colors, n_clusters = utils.hdbscan_(fitted)
    xs = fitted[:, 0]
    ys = fitted[:, 1]
    zs = fitted[:, 2]
    N = len(np.unique(sub_colors))
    labels += 1

    fig = plt.figure(num='hdbscan', figsize=(16,9))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax = fig.add_subplot(111, projection='3d')

    legend_elements = []

    for i in range(N):
        n_elements = np.unique(sub_colors, return_counts=True)[1][i]

        # For the ratios of scids in the legend
        total_nr = waves[i==labels].shape[0]/100
        per1 = round(waves[(i==labels) & (1 == icp_types)].shape[0]/total_nr, 1)
        per2 = round(waves[(i==labels) & (2 == icp_types)].shape[0]/total_nr, 1)

        legend_elements.append(Patch(facecolor=utils.default_colors[i],
                                     edgecolor='black',
                                     label='{} ({}% scp | {}% nsc)'.format(n_elements,
                                                                     per1,
                                                                     per2)))

    ax.scatter(xs, ys, zs,
               c=sub_colors,
               cmap=matplotlib.colors.ListedColormap(utils.default_colors, N=N),
               depthshade=False,
               s=25)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.legend(handles=legend_elements, loc='upper right', fontsize = 'large')
    plt.title('HDBSCAN of {}({})'
              .format(wave_type, scid))
    plt.show(block=False)
    plt.pause(0.001)

    if input('\nWant the means? (y|N)\n') != 'y':
        print('ok')
    else:
        clusters_list = []
        mean_waves = np.zeros((N, 780))
        #for i in range(N-1): # -1 because the noise is not included
        for i in range(N):
            # Mean acquiring:
            print(i)
            print(waves[i==labels].shape)
            clusters_list.append(waves[i==labels, :])
            mean_waves[i] = np.mean(waves[i==labels], axis=0)

        utils.plot_means_of_clusters(N, mean_waves,
                                     title = '{} Means ({}) (1st = noise)'.format(scid,
                                                                    decision))

        outfile = pwd + '{}_clusters_hdbscan.npy'.format(scid)
        np.save(outfile, np.asarray(clusters_list), allow_pickle=True)
        x = np.load(outfile, allow_pickle=True)
        if input('\nSatisfied? (y|N)\n') != 'y':
            raise Exception('Not satisified :(')
# ______________________________________________________________________________
else:
    print('EoF')
