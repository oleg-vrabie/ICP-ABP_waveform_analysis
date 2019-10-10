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
import hdbscan
import matplotlib
from matplotlib.patches import Patch
# =============================================================================
# Parameters
patient_nr = 37
scid = 'nsc'
wave_type = 'abp'
pwd = '/home/ov/data/stacked/'

wf = 3
scid = 'nsc'
wave_type = 'abp'
pwd = '/home/ov/data/stacked/selected/trial2/'
# =============================================================================
# Load waves:
waves = np.load(pwd + 'wf{}_{}.npy'.format(wf, scid))
# =============================================================================
# Load waves:
#stacked_waves = np.load(pwd_stacked_out)
#stacked_waves = utils.load_waves_by_pat_scid([patient_nr], scid, pwd)
#waves = utils.acces_waves_by_type(None, stacked_waves, wave_type)
#print('Stacked size: {}'.format(stacked_waves.shape))
#print('Unstacked {} waves: {}'.format(wave_type, waves.shape))
# =============================================================================
# Fit-Transforming PCA on acquired waves:
fitted = utils.pca_projections(waves, 3, svd_solver='arpack')
print('Fitted PCA: {}'.format(fitted.shape))
xs = fitted[:, 0]
ys = fitted[:, 1]
zs = fitted[:, 2]
# =============================================================================
# Gaussian Mixture Models and, if needed, additional clustering
#n_components = int(input('Type the number of components to split into: '))
n_components = 1

gmm = utils.gaussian_mixture_(fitted, waves,
                              n_components,
                              npat=patient_nr,
                              scid=scid,
                              wave_type=wave_type,
                              n_init=10)#,
                              #random_state=1)
# =============================================================================
"""
fig = plt.figure(num='Projection',figsize=(16,9))
fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, hspace=0.3)
ax = fig.add_subplot(111, projection='3d')
"""

predictions = gmm.predict(fitted)
clusters_list = []

for i in range(n_components):
    clusters_list.append(np.asarray(fitted[i==predictions]))
    print(clusters_list[i].shape)
    xs = clusters_list[i][:, 0]
    ys = clusters_list[i][:, 1]
    zs = clusters_list[i][:, 2]
    print('\t{}/{} of shape {}'.format(i+1, n_components, xs.shape))

    """
    ax.scatter(xs, ys, zs,
                c=utils.default_colors[i],
                marker="${}$".format(i+1),
                depthshade=False,
                s=60,
                label='{} elements'.format(xs.shape[0]))
    """
"""
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend()
plt.title('Projection of original {}({}) data onto first three PCs (NFPAT{})'
          .format(wave_type, scid, patient_nr))
plt.show()
"""

def hdbscan_(X,
             min_cluster_size, #= int(input('min_cluster_size: ')),
             min_samples): #= int(input('min_samples: '))):

    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                            min_samples=min_samples).fit(X)
    labels = model.labels_
    print('components_ {}'.format(labels.max()+1))
    n_clusters = labels.max()+1
    sub_colors = labels.ravel()
    return labels, sub_colors, n_clusters


#gmm_component = int(input('Component: '))
gmm_component = 1
X = np.reshape(clusters_list[0],
                       (clusters_list[0].shape[0] ,3))
print(X.shape)

results = []
#results = np.zeros((725, 2))
#ss = [i for i in range(5, 150, 5)]
#ns = [i for i in range(1, 51, 2)]
arraaa = np.zeros((725, 4))

assert results.shape[0] == len(ss) and results.shape[1] == len(ns)

for s in range(5, 150, 5):
    #print('###################################################################')
    for n in range(1, 51, 2):
        print('\nSize: {} and samples: {}'.format(s, n))
        labels, sub_colors, n_clusters = hdbscan_(X, s, n)
        #print('n_clusters = {}'.format(n_clusters))
        #print('Rest: {}'.format(np.unique(sub_colors, return_counts=True)[1][0]))
        results.append([n_clusters, np.unique(sub_colors, return_counts=True)[1][0], s, n])


# Change ONLY SIZE
results = []
for s in range(5, 80, 5):
    print('\nSize: {} and samples: {}'.format(s, 1))
    labels, sub_colors, n_clusters = hdbscan_(X, s, 1)
    print('n_clusters = {}'.format(n_clusters))
    print('Rest: {}'.format(np.unique(sub_colors, return_counts=True)[1][0]))
    #results.append([n_clusters, np.unique(sub_colors, return_counts=True)[1][0]])
    results.append([n_clusters, np.unique(sub_colors, return_counts=True)[1][0], s, 1])


# Change ONLY SAMPLES
results = []
s = 12
for n in range(0, 20, 1):
    print('\nSize: {} and samples: {}'.format(s, 2))
    labels, sub_colors, n_clusters = hdbscan_(X, s, 2)
    print('n_clusters = {}'.format(n_clusters))
    print('Rest: {}'.format(np.unique(sub_colors, return_counts=True)[1][0]))
    #results.append([n_clusters, np.unique(sub_colors, return_counts=True)[1][0]])
    results.append([n_clusters, np.unique(sub_colors, return_counts=True)[1][0], s, n])


arraaa = np.asarray(results)
print(arraaa.shape)
print(arraaa)

# do it with results as 2D array
print(result[:, 1])
boolar = result[:,1] < 2000
print(result[result[:,1] < 2000])
print(arraaa[(arraaa[:,1] < 2000) & ((arraaa[:,1] > 1000))])
print(arraaa)

plt.scatter(result[:, 0], result[:, 1])
plt.show()


# Plotting
labels, sub_colors, n_clusters = hdbscan_(X, 70, 1)

xs = clusters_list[gmm_component-1][:, 0]
ys = clusters_list[gmm_component-1][:, 1]
zs = clusters_list[gmm_component-1][:, 2]

fig = plt.figure(num='Component {}'.format(gmm_component), figsize=(16,9))
fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs,
           c=sub_colors,
           cmap=matplotlib.colors.ListedColormap(utils.default_colors, N=len(np.unique(sub_colors))),
           depthshade=False,
           s=30)

ax.set_xlabel('PC1 ({})'.format(gmm_component))
ax.set_ylabel('PC2 ({})'.format(gmm_component))
ax.set_zlabel('PC3 ({})'.format(gmm_component))
plt.title('Separate component #{} into {} clusters ({} elements)'.format(gmm_component, gmm_component, xs.shape[0]))

legend_elements = []
for i in range(len(np.unique(sub_colors))):
    legend_elements.append(Patch(facecolor=utils.default_colors[i],
                            edgecolor='black',
                            label='{} elements'.format(np.unique(sub_colors, return_counts=True)[1][i])))
legend = ax.legend(handles=legend_elements, loc='upper right')
plt.show()

