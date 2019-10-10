import scipy.io
import numpy as np
import sys
sys.path.append('/home/ov/python/py_utils')
import utils
import os, fnmatch
# #############################################################################
# Parameters
wfs = [1, 2, 3]
scid_list = ['nsc', 'scp']
pwd_stacked = '/home/ov/data/stacked/selected/trial2/'
# #############################################################################
def find(pattern, path):
    # https://stackoverflow.com/questions/1724693/find-a-file-in-python
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result
# #############################################################################
# Collecting and concatenating waves by WF and scid
for wf in wfs:
    for scid in scid_list:
        print('Waveform {} {}'.format(wf, scid))
        list = find('wf{}_{}*'.format(wf, scid), pwd_stacked)
        array = np.empty((0, 780))
        for file in list:
            #print(np.load(file).shape)
            array = np.concatenate((array, np.load(file)))
        out_path = pwd_stacked + 'wf{}_{}.npy'.format(wf, scid)
        np.save(out_path, array)
