# #############################################################################
# Script for plotting extracted ABP means
# #############################################################################
import scipy.io
import numpy as np
import sys
sys.path.append('/home/ov/python/py_utils')
import utils
import matplotlib.pyplot as plt
# =============================================================================
# Parameters
patient_nr = 7
scid = 'nsc'
wave_type = 'abp'
pwd = '/home/ov/data/stacked/'
# =============================================================================
# Load waves:
stacked_waves = utils.load_waves_by_pat_scid([patient_nr], scid, pwd, suffix='.stacked_means_icp_abp.npy')
icp_means = utils.acces_waves_by_type(None, stacked_waves, wave_type)
print('Stacked size: {}'.format(stacked_waves.shape))
print('Unstacked {} waves: {}'.format(wave_type, icp_means.shape))
# =============================================================================
if input('Satisfied? (y|N)\n') != 'y':
    raise Exception('Not satisified :(')
utils.plot_means_a4(icp_means,
                    title='PAT{} {} means ({})'.format(patient_nr, wave_type, scid),
                    orientation='landscape', side1=3, side2=4,
                    savefig=True,
                    path='/home/ov/data/processing/extracted_means/')
# =============================================================================

# =============================================================================
