# #############################################################################
# Script for ...
# #############################################################################
import scipy.io
import numpy as np
import sys
sys.path.append('/home/ov/python/py_utils')
import utils
import matplotlib.pyplot as plt
# =============================================================================
# Parameters
patient_nr = [7, 10, 11, 12, 18, 23, 37]
scid = 'scp'
wave_type = 'abp'
pwd = '/home/ov/data/stacked/'
# =============================================================================
# Load waves:
#stacked_waves = np.load(pwd_stacked_out)
stacked_waves = utils.load_waves_by_pat_scid(patient_nr, scid, pwd, suffix='.stacked_means_icp_abp.npy')
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

utils.plot_3d_mod(xs, ys, zs,
                    title='PCA projection of {} waves of PAT{} ({})'.format(wave_type,
                                                                            patient_nr,
                                                                            scid),
                    s=10,
                    edge = 'navy',
                    xlabel = 'PC1',
                    ylabel = 'PC2',
                    zlabel = 'PC3',
                    add_info = 'PAT{} ({})'.format(patient_nr, scid))

if input('\nContinue to clustering? (y|N)\n') != 'y':
    raise Exception('Not satisified :(')
# =============================================================================
# Gaussian Mixture Models and, if needed, additional clustering
n_components = int(input('Type the number of components to split into: '))

cluster_means, clusters, ns, ks, clusters_list = utils.gaussian_mixture_pca_projections(
                                                fitted, waves, n_components,
                                                npat=patient_nr, scid=scid,
                                                wave_type=wave_type)

if input('\nContinue to stacking? (y|N)\n') != 'y':
    raise Exception('Not satisified :(')
# =============================================================================
# DO I NEED TO SAVE STACKED ABP AND ICP MEANS?
assert len(cluster_means) == len(clusters_list)

counter_waves = utils.acces_waves_by_type(None, stacked_waves, 'icp')
icp_means = np.zeros((len(clusters_list), 780))

for i in range(len(clusters_list)):
    print(clusters_list[i].shape)
    mean = np.mean(counter_waves[clusters_list[i], :], axis=0)
    icp_means[i] = mean

stacked_means = np.stack((icp_means, cluster_means))
print('Stacked shape: {}'.format(stacked_means.shape))

if input('\nContinue to saving? (y|N)\n') != 'y':
    raise Exception('Not satisified :(')
# =============================================================================
# Saving extracted means
pwd_means = pwd + 'nfpat' + str(patient_nr) + scid + '.stacked_means_icp_abp'
np.save(pwd_means, stacked_means)
