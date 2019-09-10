import scipy.io
import numpy as np
import sys
sys.path.append('/home/ov/python/py_utils')
import utils
# Normalization:
from sklearn.preprocessing import normalize

# Parameters
patient_list = [7, 10, 11, 12, 18, 23, 37]
scid_list = ['nsc', 'scp']
pwd_stacked = '/home/ov/data/stacked/'

for patient_nr in patient_list:
    for scid in scid_list:
        print('Patient {} {}'.format(patient_nr, scid))
        # Load ICP
        pwd_icp = '/home/ov/icp_project/WAVEFORMETECTOR/NFPAT' + str(patient_nr) + '/SCPSEGSHIGH/output/'
        wave_type = 'icp'
        pwd_icp_mat = pwd_icp + 'nfpat' + str(patient_nr) + '.' + wave_type + '.' + scid + 'seg.waves.mat'
        icp_waves = scipy.io.loadmat(pwd_icp_mat)['waves_mat']
        icp_waves = normalize(icp_waves, norm='max')
        print(icp_waves.shape)

        # Load ABP
        wave_type = 'abp'
        pwd_abp = '/home/ov/data/WAVESEGMENTS/abp/output/'
        pwd_abp_mat = pwd_abp + 'nfpat' + str(patient_nr) + '.' + wave_type + '.' + scid + 'seg.waves.mat'
        abp_waves = scipy.io.loadmat(pwd_abp_mat)['waves_mat']
        abp_waves = normalize(abp_waves, norm='max')
        print(abp_waves.shape)

        # Stacking ICP and ABP together
        stacked_icp_abp = np.stack((icp_waves, abp_waves))
        print('Stacked: {}'.format(stacked_icp_abp.shape))
        pwd_stacked_out = pwd_stacked + 'nfpat' + str(patient_nr) + scid + '.stacked_icp_abp'
        np.save(pwd_stacked_out, stacked_icp_abp)
