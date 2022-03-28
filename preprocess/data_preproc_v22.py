import nibabel as nib
import numpy as np
from preprocess.normalization import normalize_mri
from scipy import spatial
import collections
import h5py
import pickle

class imagepatches(object):

    def __init__(self, fname, mask, label=None,
                 gm=150, wm=250, csf=10, num_classes=4, pad = 3, masklabel=False,k_t2=4, k_t2_init=None,
                 channels=None, normalize=True, normfactors = None,
                 setbounds = False, bounds=(),
                 fnoutput='data.h5', dataNum = 0, pkl = False):

        self.pad = pad
        self.masklabel = masklabel
        self.norm = normalize
        self.fname = fname
        self.fname_mask = mask
        self.dataNum = dataNum
        self.fnoutput = fnoutput
        self.pkl = pkl
        if normfactors:
            self.normfactors = normfactors
        else:
            self.normfactors = None

        self.k_t2 = k_t2
        self.k_t2_init=k_t2_init
        self.nii = nib.load(fname)
        self.data = self.nii.get_fdata()
        self.dataOrigShape = self.data.shape

        self.indices = []
        self.indices_upsampled = []

        self.nii3 = nib.load(mask)
        self.mask = self.nii3.get_fdata()

        self.gm = gm
        self.wm = wm
        self.csf = csf
        if label:
            self.label = nib.load(label).get_fdata()
            self.label[self.label == 0] = 4
            self.label[self.label == wm] = 0
            self.label[self.label == gm] = 1
            self.label[self.label == csf] = 2
        else:
            self.label = label
        self.num_classes = num_classes

        self.indices_upsampled =[]

        self.preproc()
        if setbounds is False:
            self.get_bounds()
        else:
            (self.xpos, self.xpos_end) = bounds[0]
            (self.ypos, self.ypos_end) = bounds[1]
            (self.zpos, self.zpos_end) = bounds[2]

        if self.norm:
            self.normalize()
        self.create_data_struct()

    def preproc(self):
        self.mask[self.mask >0] =1
        self.dataUpsampledShape = self.data.shape

    def normalize(self):
        if self.normfactors:
            self.mean, self.std = self.normfactors
        else:
            self.mean, self.std = normalize_mri(self.data, self.mask, self.k_t2, self.k_t2_init)
        idxs = np.where(self.data > 0)
        self.data = self.data.astype(np.float64)
        self.data[idxs] -= self.mean
        self.data[idxs] /= self.std

    ## get bounds
    def get_bounds(self):
        for i in range(self.mask.shape[0]):
            if np.sum(self.mask[i, :,:]) != 0:
                self.xpos = i
                break

        for i in range(self.mask.shape[0]-1, 0, -1):
            if np.sum(self.mask[i, :,:]) != 0:
                self.xpos_end = i+1
                break

        for i in range(self.mask.shape[1]):
            if np.sum(self.mask[:, i,:]) != 0:
                self.ypos = i
                break

        for i in range(self.mask.shape[1]-1, 0, -1):
            if np.sum(self.mask[:, i,:]) != 0:
                self.ypos_end = i+1
                break

        for i in range(self.mask.shape[2]):
            if np.sum(self.mask[:, :,i]) != 0:
                self.zpos = i
                break

        for i in range(self.mask.shape[2]-1, 0, -1):
            if np.sum(self.mask[:, :,i]) != 0:
                self.zpos_end = i+1
                break

    def create_data_struct(self):
        data = {'data':self.data,
                'targets': self.label,
                'origsize': self.dataOrigShape,
                'bounds': [[self.xpos, self.xpos_end],
                           [self.ypos, self.ypos_end],
                           [self.zpos, self.zpos_end]],
                'datasetNum' : self.dataNum,
                'mask' : self.mask
                }
        if self.pkl:
            file_obj = open(self.fnoutput+'.obj', 'wb')
            pickle.dump(data, file_obj, protocol=4)
        else:
            chunks=None
            with h5py.File(self.fnoutput + '.h5', "w") as f:
                f.create_dataset('data', data=data['data'], chunks=chunks)
                f.create_dataset('targets', data=self.label, chunks=chunks)
                f.create_dataset('mask', data=self.mask, chunks=chunks)
                f.attrs['bounds'] = data['bounds']
                f.attrs['datasetNum'] = self.dataNum
                f.attrs['origsize'] = self.dataOrigShape

        print('Dataset files generated.')