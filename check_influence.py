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
# =============================================================================
# Load clusters:
clusters_nsc = np.load(pwd + 'nsc_clusters_hdbscan.npy', allow_pickle = True)
clusters_scp = np.load(pwd + 'scp_clusters_hdbscan.npy', allow_pickle = True)
print(clusters_scp[5].shape)
print(clusters_nsc[1].shape)

fitted_scp = utils.pca_projections(clusters_scp[5], 3, svd_solver='arpack')
fitted_nsc = utils.pca_projections(clusters_nsc[1], 3, svd_solver='arpack')

xs_list = []
ys_list = []
zs_list = []

xs_list.append(fitted_scp[:, 0])
xs_list.append(fitted_nsc[:, 0])
ys_list.append(fitted_scp[:, 1])
ys_list.append(fitted_nsc[:, 1])
zs_list.append(fitted_scp[:, 2])
zs_list.append(fitted_nsc[:, 2])

# =============================================================================
# Plot PCA proj. of ICP corresponding to each wf:
title = 'Projection of ICP scp&nsc waves from two clusters with same input wfs'
utils.stack_plot(xs_list, ys_list, zs_list,
               label_prefix = 'scid',
               title = title,
               s = 40)
# =============================================================================
# Fit PCA and plot by scid
fitted = utils.pca_projections(waves, 3, svd_solver='arpack')
print('Fitted PCA: {}'.format(fitted.shape))
xs = fitted[:, 0]
ys = fitted[:, 1]
zs = fitted[:, 2]
figu, axu = utils.plot_3d_mod(xs, ys, zs,
                    title='all {} icp waves from selected wfs'.format(scid),
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
    labels, sub_colors, n_clusters = utils.hdbscan_(fitted)
    xs = fitted[:, 0]
    ys = fitted[:, 1]
    zs = fitted[:, 2]
    N = len(np.unique(sub_colors))
    labels += 1

    #print(waves[2 == np.where(wf_types == 1)].shape)
    #print(waves[(1==labels) & (1 == wf_types)].shape)

    fig = plt.figure(num='hdbscan', figsize=(16,9))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs,
               c=sub_colors,
               cmap=matplotlib.colors.ListedColormap(utils.default_colors, N=N),
               depthshade=False,
               s=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.title('HDBSCAN ({})'.format(scid))

    legend_elements = []

    from matplotlib.patches import Patch

    for i in range(N):
        n_elements = np.unique(sub_colors, return_counts=True)[1][i]

        # For the ratios of wfs in the legend
        total_nr = waves[i==labels].shape[0]/100
        per1 = round(waves[(i==labels) & (1 == wf_types)].shape[0]/total_nr, 1)
        per2 = round(waves[(i==labels) & (2 == wf_types)].shape[0]/total_nr, 1)
        per3 = round(waves[(i==labels) & (3 == wf_types)].shape[0]/total_nr, 1)

        legend_elements.append(Patch(facecolor=utils.default_colors[i],
                                     edgecolor='black',
                                     label='{} ({}%|{}%|{}%)'.format(n_elements,
                                                                     per1,
                                                                     per2,
                                                                     per3)))

    legend = ax.legend(handles=legend_elements, loc='upper right')
    plt.show(block=False)
    plt.pause(0.001)

    if input('\nWant the means? (y|N)\n') != 'y':
        print('ok')
    else:
        clusters_list = []
        mean_waves = np.zeros((N, 780))
        for i in range(N-1): # -1 because the noise is not included
            # Mean acquiring:
            print(waves[i==labels].shape)
            clusters_list.append(waves[i==labels, :])
            mean_waves[i] = np.mean(waves[i==labels], axis=0)

        utils.plot_means_of_clusters(N-1, mean_waves,
                                     title = '{} Means ({})'.format(scid,
                                                                    decision))

        outfile = pwd + '{}_clusters_hdbscan.npy'.format(scid)
        np.save(outfile, np.asarray(clusters_list), allow_pickle=True)
        x = np.load(outfile, allow_pickle=True)
        if input('\nSatisfied? (y|N)\n') != 'y':
            raise Exception('Not satisified :(')
# ______________________________________________________________________________
else:
    print('EoF')