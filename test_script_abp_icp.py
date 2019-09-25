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

# #############################################################################
def load_waves_by_pat_scid(patids, scid, pwd):

    i = 0
    for patid in patids:
        pwd_stacked = pwd + 'nfpat' + str(patid) + scid + '.stacked_icp_abp.npy'
        if i == 0:
            stacked_waves = np.load(pwd_stacked)
        else:
            stacked_waves = np.concatenate((stacked_waves, np.load(pwd_stacked)), axis = 1)
        print(stacked_waves.shape)
        i += 1
    return stacked_waves

def plot_pca_by_pat_scid(patids, scid, ):
    fig = plt.figure(num='pca by pat and scid',figsize=(16,9))
    fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.3)
    ax = fig.add_subplot(111, projection='3d')
# #############################################################################



#patids = [7, 10, 11, 12, 18, 23, 37]
patids = [7, 10, 37]
#patids = [11, 12, 18, 23]
patient_nr = patids
scid = 'scp'
pwd_stacked = '/home/ov/data/stacked/'
#stacked_waves = load_waves_by_pat_scid(patids, scid, pwd_stacked)
wave_type = 'abp'




fig = plt.figure(num='ih',figsize=(16,9))
fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.3)
ax = fig.add_subplot(111, projection='3d')

i = 0
pwd = pwd_stacked
for patid in patids:
    #pwd_stacked = pwd + 'nfpat' + str(patid) + scid + '.stacked_icp_abp.npy'
    #stacked_by_pat = np.load(pwd_stacked)
    stacked_by_pat = load_waves_by_pat_scid([patid], scid, pwd)
    waves_by_pat = utils.acces_waves_by_type(None, stacked_by_pat, wave_type)
    fitted = utils.pca_projections(waves_by_pat, 3, svd_solver='arpack')
    print('Fitted PCA: {}'.format(fitted.shape))
    xs = fitted[:, 0]
    ys = fitted[:, 1]
    zs = fitted[:, 2]
    fig, ax = utils.plot_3d_mod(xs, ys, zs,
                                title='',
                                s=1,
                                edge = 'navy',#utils.default_colors[i],
                                fig = fig,
                                ax = ax,
                                xlabel = 'PC1',
                                ylabel = 'PC2',
                                zlabel = 'PC3',
                                add_info = 'PAT{} ({})'.format(patid, scid),
                                pause_time=0.1)
    i += 1
plt.suptitle('PCA projection of entire {} data'.format(scid))
plt.show(block=False)
plt.pause(0.001)
"""
i = 0
pwd = pwd_stacked
face = 'none'
for patid in patids:
    pwd_stacked = pwd + 'nfpat' + str(patid) + scid + '.stacked_icp_abp.npy'
    stacked_by_pat = np.load(pwd_stacked)
    waves_by_pat = utils.acces_waves_by_type(None, stacked_by_pat, wave_type)
    fitted = utils.pca_projections(waves_by_pat, 3, svd_solver='arpack')
    print('Fitted PCA: {}'.format(fitted.shape))
    xs = fitted[:, 0]
    ys = fitted[:, 1]
    zs = fitted[:, 2]
    edge = utils.default_colors[i]

    ax.scatter(xs, ys, zs,
               depthshade=False,
               edgecolors=edge,
               facecolors=face,
               s=1,
               label='{} elements of PAT{}'.format(xs.shape[0], patid))

    i += 1

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.title('title')
#plt.show()
plt.show(block=False)
plt.pause(0.001)
"""

if input('Satisfied? (y|N)\n') != 'y':
    raise Exception('Not satisified :(')
"""
# =============================================================================
# Parameters
#patient_nr = 7#[7, 10, 11, 12, 18, 23, 37]
#scid = 'scp'
wave_type = 'abp'
pwd_stacked = '/home/ov/data/stacked/'
pwd_stacked_out = pwd_stacked + 'nfpat' + str(patient_nr) + scid + '.stacked_icp_abp.npy'
# =============================================================================
# Load waves:
stacked_waves = np.load(pwd_stacked_out)
"""
stacked_waves = load_waves_by_pat_scid(patids, scid, pwd)

waves = utils.acces_waves_by_type(None, stacked_waves, wave_type)
print('Stacked size: {}'.format(stacked_waves.shape))
print('Unstacked {} waves: {}'.format(wave_type, waves.shape))
# =============================================================================
# Fit-Transforming PCA on acquired waves:
fitted = utils.pca_projections(waves, 3, svd_solver='arpack')
print('Fitted PCA: {}'.format(fitted.shape))
"""
xs = fitted[:, 0]
ys = fitted[:, 1]
zs = fitted[:, 2]

utils.plot_3d(xs, ys, zs,
              title='PCA projection of {} waves of PAT{} ({})'.format(wave_type,
                                                                      patient_nr,
                                                                      scid),
              xlabel='PC1',
              ylabel='PC2',
              zlabel='PC3',
              s=1)
"""
# =============================================================================
# Gaussian Mixture Models and, if needed, additional clustering
n_components = int(input('Type the number of components to split into: '))

fitted_copy = fitted

cluster_means, clusters, ns, ks, clusters_list = utils.gaussian_mixture_pca_projections(
                                                fitted, waves, n_components,
                                                npat=patient_nr, scid=scid,
                                                wave_type=wave_type)
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


print(input('To exit, press any key!\n'))

"""
    # Gaussian Mixture Models and, if needed, additional clustering
    n_components_ = int(input('Type the number of components to split into: '))

    cluster_means_, clusters_, ns_, ks_, clusters_list_ = utils.gaussian_mixture_pca_projections(
                                          fitted_, counter_clusters, n_components_)
    print(len(clusters_list[i]))
"""
