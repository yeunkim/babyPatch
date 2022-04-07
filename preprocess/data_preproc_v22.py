import nibabel as nib
import numpy as np
from preprocess.normalization import normalize_mri
from scipy import spatial
import collections
import h5py
import pickle
from scipy.ndimage import affine_transform
from collections import namedtuple

## modification: pad image for skull stripping

class imagepatches(object):

    def __init__(self, fname, mask=None, label=None,
                 gm=150, wm=250, csf=10, num_classes=4, pad = 3, masklabel=False,k_t2=4, k_t2_init=None,
                 channels=None, normalize=True, normfactors = None,
                 setbounds = False, bounds=(), skullstrip=True, ram = False,
                 fnoutput='data.h5', dataNum = 0, pkl = False):

        self.pad = pad
        self.masklabel = masklabel
        self.norm = normalize
        self.fname = fname
        self.fname_mask = mask
        self.dataNum = dataNum
        self.fnoutput = fnoutput
        self.pkl = pkl
        self.ram = ram
        if normfactors:
            self.normfactors = normfactors
        else:
            self.normfactors = None

        self.k_t2 = k_t2
        self.k_t2_init=k_t2_init
        self.nii = nib.load(fname)
        self.data = self.nii.get_fdata()
        self.dataaff = self.nii._affine
        # diff = self.dataaff[:3,:3] - np.eye(3)
        # if np.count_nonzero(diff) > 3:
        #     diff[diff !=0] = 1
        #     axes = np.where(np.sum(diff, axis=1) >1)[0]
        #     if not np.any(axes == 0):
        #         self.data = self.data.transpose((0, 2, 1))
        #     if not np.any(axes == 1):
        #         self.data = self.data.transpose((2,1, 0))
        #     if not np.any(axes == 2):
        #         self.data = self.data.transpose((1,0, 2))
        self.dataOrigShape = self.data.shape

        self.indices = []
        self.indices_upsampled = []
        self.skullstrip = skullstrip

        self.gm = gm
        self.wm = wm
        self.csf = csf
        if label:
            self.labelnii = nib.load(label)
            self.labelaff = self.labelnii._affine
            self.label = self.labelnii.get_fdata()
            if self.skullstrip:
                self.label[self.label > 0] = 1
            else:
                self.nii3 = nib.load(mask)
                self.maskaff = self.nii3._affine
                self.mask = self.nii3.get_fdata()
                self.label[self.label == 0] = 4
                self.label[self.label == wm] = 0
                self.label[self.label == gm] = 1
                self.label[self.label == csf] = 2
        else:
            self.label = label
        self.preproc()
        if self.skullstrip:
            self.run_pad()
        if self.norm:
            self.normalize()
        self.num_classes = num_classes

        self.indices_upsampled =[]

        if setbounds is False:
            self.get_bounds()
        else:
            (self.xpos, self.xpos_end) = bounds[0]
            (self.ypos, self.ypos_end) = bounds[1]
            (self.zpos, self.zpos_end) = bounds[2]

        self.create_data_struct()

    def preproc(self):
        self.dataUpsampledShape = self.data.shape
        if self.skullstrip:
            self.mask = np.zeros((2*self.pad+self.dataOrigShape[0],
                                 2*self.pad+self.dataOrigShape[1],
                                 2*self.pad+self.dataOrigShape[2]), dtype=np.int)
            self.mask[self.pad:-self.pad, self.pad:-self.pad, self.pad:-self.pad] = 1
        self.mask[self.mask >0] =1

        # self.mask = affine_transform(self.mask, self.maskaff, order=0)
        # self.label = affine_transform(self.label, self.labelaff, order=0)
        # self.data = affine_transform(self.data, self.dataaff)
        # print('stop')
    def normalize(self):
        if self.normfactors:
            self.mean, self.std = self.normfactors
        else:
            self.mean, self.std = normalize_mri(self.data, self.mask, self.k_t2, self.k_t2_init)
        idxs = np.where(self.data > 0)
        self.data = self.data.astype(np.float64)
        self.data[idxs] -= self.mean
        self.data[idxs] /= self.std

    def run_pad(self):
        if len(self.data.shape) == 3:
            self.data = np.pad(self.data,self.pad)
        if len(self.data.shape) == 4:
            self.data = np.pad(self.data,((self.pad,self.pad),(self.pad,self.pad),(self.pad,self.pad),(0,0)))

        self.label = np.pad(self.label, self.pad)
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
            print('Dataset files generated.')
        elif self.ram:
            dataset = namedtuple('dataset', ('data', 'targets', 'mask', 'bounds', 'origsize'))
            self.dataset = dataset(data['data'], data['targets'], data['mask'], data['bounds'], data['origsize'])
        elif not self.pkl:
            chunks=None
            with h5py.File(self.fnoutput + '.h5', "w") as f:
                f.create_dataset('data', data=data['data'], chunks=chunks)
                f.create_dataset('targets', data=self.label, chunks=chunks)
                f.create_dataset('mask', data=self.mask, chunks=chunks)
                f.attrs['bounds'] = data['bounds']
                f.attrs['datasetNum'] = self.dataNum
                f.attrs['origsize'] = self.dataOrigShape
            print('Dataset files generated.')

    def return_data_struct(self):
        return self.dataset