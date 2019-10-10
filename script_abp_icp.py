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
patient_nr = 10
scid = 'scp'
wave_type = 'abp'
pwd = '/home/ov/data/stacked/'
# =============================================================================
# Load waves:
#stacked_waves = np.load(pwd_stacked_out)
stacked_waves = utils.load_waves_by_pat_scid([patient_nr], scid, pwd)
waves = utils.acces_waves_by_type(None, stacked_waves, wave_type)
print('Stacked size: {}'.format(stacked_waves.shape))
print('Unstacked {} waves: {}'.format(wave_type, waves.shape))
# =============================================================================
# Fit-Transforming PCA on acquired waves:
fitted = utils.pca_projections(waves, 3, svd_solver='arpack')
print('Fitted PCA: {}'.format(fitted.shape))
xs = fitted[:, 0]
ys = fitted[:, 1]
zs = fitted[:, 2]
"""
utils.plot_3d_mod(xs, ys, zs,
                    title='',
                    s=1,
                    edge = 'navy',
                    xlabel = 'PC1',
                    ylabel = 'PC2',
                    zlabel = 'PC3',
                    add_info = 'PAT{} ({})'.format(patient_nr, scid))
if input('Satisfied? (y|N)\n') != 'y':
    raise Exception('Not satisified :(')
"""
# =============================================================================
# Gaussian Mixture Models and, if needed, additional clustering
n_components = int(input('Type the number of components to split into: '))

fitted_copy = fitted

cluster_means, clusters, ns, ks, clusters_list = utils.gaussian_mixture_pca_projections(
                                                fitted, waves, n_components,
                                                npat=patient_nr, scid=scid,
                                                wave_type=wave_type,
                                                n_init=1,
                                                random_state=120000)
# =============================================================================
# Plot random ABP waves and their ICP counterparts
"""
utils.plot_random_wave_samples(waves, wave_type='abp', seg_type=scid, rand_list=clusters_list[-1])
icp_waves = utils.acces_waves_by_type(None, stacked_waves, 'icp')
utils.plot_random_wave_samples(icp_waves, wave_type='icp', seg_type=scid, rand_list=clusters_list[-1])
"""
# =============================================================================
# Fit-Transforming PCA on acquired waves:
print('\n\nAnalysis of corresponding counterparts:')
wave_type = 'icp'
counter_waves = utils.acces_waves_by_type(None, stacked_waves, wave_type)
fitted_all = utils.pca_projections(counter_waves, 3, svd_solver='arpack')
# =============================================================================
# Plot a 2x2 grid to compare the PCA projections and corresponding means of
# ABP and ICP waves
print(np.array_equal(fitted_copy, fitted))

for i in range(len(clusters_list)):
    #counter_clusters = counter_waves[clusters_list[i], :]
    fitted_ = fitted_all[clusters_list[i], :]   # counter_waves PCA projections
    fitted_abp = fitted[clusters_list[i], :]
    print(counter_waves[clusters_list[i], :].shape)
    icp_mean = np.mean(counter_waves[clusters_list[i], :], axis=0)
    #abp_mean = cluster_means[i]
    abp_mean = np.mean(clusters[i], axis=0)

    # IS IT CORRECT TO DO PCA ON A PART OF DATA???
    #fitted_ = utils.pca_projections(counter_clusters, 3, svd_solver='arpack')
    print('Fitted PCA: {}'.format(fitted_.shape))
    xs_a = fitted_abp[:, 0]
    ys_a = fitted_abp[:, 1]
    zs_a = fitted_abp[:, 2]
    xs_i = fitted_[:, 0]
    ys_i = fitted_[:, 1]
    zs_i = fitted_[:, 2]

    utils.plot_two_by_two(xs_a, ys_a, zs_a,
                        xs_i, ys_i, zs_i,
                        values3 = abp_mean,
                        values4 = icp_mean,
                        window_title='{}'.format(i+1),#clusters[i].shape),
                        suptitle='Comparison of ABP cluster #{} and corresponding ICP waves'.format(i+1),
                        title1='ABP #{}'.format(i+1),
                        title2='ICP #{}'.format(i+1),
                        title3='ABP mean #{}'.format(i+1),
                        title4='ICP mean #{}'.format(i+1))

# =============================================================================
# Saving ICP waves corresponding to selected ABP means
if input('\n\nCluster selection? (y|N)\n') == 'y':
    pwd = pwd + 'selected/trial2/'
    for wf in range(1, 4):
        list = [int(x) for x in input('\nGive the index of wf{}(s) separated by space: '.format(wf)).split()]
        print('You have selected: {}'.format(list))
        sel_indxx = []
        for i in list:
            sel_indxx.extend(clusters_list[i-1])
        print('Extracted (total wf{}s): {}'.format(wf,len(sel_indxx)))
        selected_i = counter_waves[sel_indxx]
        #filename = str(input('Give a filename to save wf{}: '.format(wf)))
        filename = 'wf{}_{}_pat{}.npy'.format(wf, scid, patient_nr)
        if sel_indxx != []:
            np.save(pwd + filename, selected_i)


print(input('To exit, press any key!'))

"""
    # Gaussian Mixture Models and, if needed, additional clustering
    n_components_ = int(input('Type the number of components to split into: '))

    cluster_means_, clusters_, ns_, ks_, clusters_list_ = utils.gaussian_mixture_pca_projections(
                                          fitted_, counter_clusters, n_components_)
    print(len(clusters_list[i]))
"""
