from sklearn.decomposition import PCA as PCA
import scipy.io
import numpy as np
import inspect
import sys
sys.path.append('/home/ov/python/py_utils')
import utils

# Parameters:
#patient_list = [7, 10, 11, 12, 18, 23, 37]
#scid_list = ['scp']

pwd = '/home/ov/preprocessed_waves/'

pwd_concat = pwd + 'entire_nsc_data.npy'    # nsc
#pwd_concat = pwd + 'entire_scp_data.npy'    # scp
concatenated_waves = np.load(pwd_concat)

print(concatenated_waves.shape)
print(concatenated_waves.shape)

# Normalization:
from sklearn.preprocessing import normalize
concatenated_waves = normalize(concatenated_waves, norm='max')

# Fit-Transforming PCA on acquired waves:
fitted = utils.pca_projections(concatenated_waves, 3, svd_solver='arpack')
print(fitted.shape)

xs = fitted[:, 0]
ys = fitted[:, 1]
zs = fitted[:, 2]

scid = 'nsc'

utils.plot_3d(xs, ys, zs,
              title='PCA projection'.format(scid),
              xlabel='PC1',
              ylabel='PC2',
              zlabel='PC3',
              s=1)

# Gaussian Mixture Models and, if needed, additional clustering
n_components = int(input('Type the number of components to split into: '))

cluster_means, clusters, ns, ks = utils.gaussian_mixture_pca_projections(
                                     fitted, concatenated_waves,
                                     n_components)


"""
clusters = np.asarray(clusters)

# Cleaning (only if additional clustering was needed):
if ns is not None:
    print('\nCleaning...')
    x = cluster_means.shape[0]

    # Means:
    cluster_means = np.delete(cluster_means, ns, axis=0)
    # Clusters:
    clusters = np.delete(clusters, ns, axis=0)

    assert cluster_means.shape[0] == clusters.shape[0]
    print('{} cluster(s) removed!'.format(x - cluster_means.shape[0]))

# Save extracted clusters and their means:
pwd_extracted = pwd + 'extracted/' +'nfpat' + str(patient_nr) + '.icp.normalized.' + scid
pwd_means = pwd_extracted + 'normalized_means'
pwd_clusters = pwd_extracted + '_clusters'
assert cluster_means.shape[0] == clusters.shape[0]
np.savetxt(pwd_means, cluster_means)
np.save(pwd_clusters, clusters)

print('\nMeans and clusters saved as: \n {} in {}\n {} in {}'.format(cluster_means.shape,
                                                                     pwd_means,
                                                                     clusters.shape,
                                                                     pwd_clusters))

"""
