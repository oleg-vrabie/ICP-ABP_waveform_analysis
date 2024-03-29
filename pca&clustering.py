from sklearn.decomposition import PCA as PCA
import scipy.io
import numpy as np
import inspect
import sys
sys.path.append('/home/ov/python/py_utils')
import utils

# Parameters:
patient_nr = 7
scid = 'nsc'
pwd = '/home/ov/preprocessed_waves/'

# Acquiring waves:
if patient_nr != 7:
    pwd_mat = pwd + 'nfpat' + str(patient_nr) + '.icp.' + scid + 'seg.waves.mat'
    concatenated_waves = scipy.io.loadmat(pwd_mat)['waves_mat']
else:
    pwd_mat1 = pwd + 'nfpat' + str(patient_nr) + '.icp.' + scid + 'seg.waves.mat'
    pwd_mat2 = pwd + 'nfpat' + str(patient_nr) + '.testicp.' + scid + 'seg.waves.mat'
    concatenated_waves = scipy.io.loadmat(pwd_mat1)['waves_mat']
    concatenated_waves = np.concatenate((concatenated_waves,
                                         scipy.io.loadmat(pwd_mat2)['waves_mat']),
                                         axis = 0)

# Normalization:
from sklearn.preprocessing import normalize
concatenated_waves = normalize(concatenated_waves, norm='max')

print(concatenated_waves.shape)

# Fit-Transforming PCA on acquired waves:
fitted = utils.pca_projections(concatenated_waves, 3, svd_solver='arpack')
print(fitted.shape)

xs = fitted[:, 0]
ys = fitted[:, 1]
zs = fitted[:, 2]

utils.plot_3d(xs, ys, zs,
              title='NFPAT{} {} PCA projection'.format(patient_nr, scid),
              xlabel='PC1',
              ylabel='PC2',
              zlabel='PC3',
              s=1)

# Gaussian Mixture Models and, if needed, additional clustering
n_components = int(input('Type the number of components to split into: '))

cluster_means, clusters, ns, ks = utils.gaussian_mixture_pca_projections(
                                     fitted, concatenated_waves,
                                     n_components, patient_nr, scid,
                                     with_mahal=False)

clusters = np.asarray(clusters)

if input('Continue with cleaning and saving? ') != 'yes':
    import sys
    sys.exit()
else:
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
