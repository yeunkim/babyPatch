import nibabel as nib
import numpy as np
from normalization import normalize_mri
import pickle
import h5py

class imagepatches(object):

    def __init__(self, fname, mask, label, fnoutput,
                 gm=150, wm=250, csf=10, num_classes=4, pad = 3, masklabel=False,
                 fname_t1=None, k_t2=4, k_t2_init=None, pickle = False,
                 k_t1=2, k_t1_init=None, normalize = True, threedim=False, channels=None, coords=None,
                 origcoords = None):

        self.pad = pad
        self.pickle = pickle
        self.masklabel = masklabel

        self.fname = fname
        self.fname_t1 = fname_t1
        self.fname_mask = mask
        self.fnoutput = fnoutput

        self.k_t2 = k_t2
        self.k_t1 = k_t1
        self.k_t2_init=k_t2_init
        if k_t2_init is None:
            self.k_t2_init = [500,100,1000,1500]
        self.k_t1_init = k_t1_init
        if k_t1_init is None:
            self.k_t1_init = [500,250]

        self.nii = nib.load(fname)

        self.data = self.nii.get_data()
        self.dataOrigShape = self.data.shape

        self.indices = []
        self.indices_upsampled = []

        self.nii3 = nib.load(mask)
        self.mask = self.nii3.get_data()

        self.gm = gm
        self.wm = wm
        self.csf = csf
        self.label = nib.load(label).get_data()

        self.label[self.label == 255] = 1
        self.label[self.label == 0] = 4
        self.label[self.label == wm] = 0
        self.label[self.label == gm] = 1
        self.label[self.label == csf] = 2
        self.num_classes = num_classes

        self.prep()
        self.get_bounds()

        if normalize:
            self.normalize()
        self.create_data_struct()

    def prep(self):
        self.mask[self.mask >0] =1
        self.data[ self.mask == 0 ]= 0
        self.origsize = self.data.shape

    def normalize(self):
        self.mean, self.std = normalize_mri(self.data, self.mask, self.k_t2, self.k_t2_init)
        idxs = np.where(self.data > 0)
        self.data = self.data.astype(np.float64)
        self.data[idxs] -= self.mean
        self.data[idxs] /= self.std


    ## get bounds
    def get_bounds(self):
        for i in range(self.mask.shape[0]):
            if np.sum(self.mask[i, :,:]) > 0:
                self.xpos = i
                break

        for i in range(self.mask.shape[0]-1, 0, -1):
            if np.sum(self.mask[i, :,:]) > 0:
                self.xpos_end = i-1
                break

        for i in range(self.mask.shape[1]):
            if np.sum(self.mask[:, i,:]) > 0:
                self.ypos = i
                break

        for i in range(self.mask.shape[1]-1, 0, -1):
            if np.sum(self.mask[:, i,:]) > 0:
                self.ypos_end = i-1
                break

        for i in range(self.mask.shape[2]):
            if np.sum(self.mask[:, :,i]) > 0:
                self.zpos = i
                break

        for i in range(self.mask.shape[2]-1, 0, -1):
            if np.sum(self.mask[:, :,i]) > 0:
                self.zpos_end = i-1
                break


    def create_data_struct(self):
        ### test
        # nonzeros = np.count_nonzero(self.X)

        xpos = self.xpos - (self.pad + 1)
        xposend = self.xpos_end + self.pad + 2
        ypos = self.ypos - (self.pad +1)
        yposend = self.ypos_end + self.pad + 2
        zpos = self.zpos - (self.pad +1)
        zposend = self.zpos_end + self.pad + 2

        if self.masklabel:
            self.indices = np.where((self.mask.ravel() > 0) & (self.label.ravel() != 4))[0]
        else:
            self.indices = np.where(self.mask.ravel() >0)[0]

        tmpmask = self.mask[xpos:xposend, ypos:yposend, zpos:zposend]
        tmplabel = self.label[xpos:xposend, ypos:yposend, zpos:zposend]
        if self.masklabel:
            self.cropped_indices = np.where((tmpmask.ravel() > 0) & (tmplabel.ravel() != 4))[0]
        else:
            self.cropped_indices  = np.where(tmpmask.ravel() >0)[0]

        assert np.sum(self.mask) == np.sum(tmpmask)

        # nonzeros = len(self.indices)

        # self.X5 = self.X5[0:nonzeros]
        # self.neighbors = self.neighbors[0:nonzeros]
        # self.neighbors_z = self.neighbors_z[0:nonzeros]
        # self.neighbors_y = self.neighbors_y[0:nonzeros]

        with h5py.File(self.fnoutput+'.h5', "w") as f:
            # g1 = f.create_group(self.groupname)
            f.create_dataset('data', data=self.data[xpos:xposend, ypos:yposend, zpos:zposend])
            f.create_dataset('targets', data=tmplabel)
            f.create_dataset('indices', data=self.indices)
            f.create_dataset('cropped_indices', data=self.cropped_indices)
            f.attrs['origsize'] = self.dataOrigShape

        if self.pickle:
            data = {'data':self.data[xpos:xposend, ypos:yposend, zpos:zposend],
                    'targets': tmplabel,
                    'indices': self.indices,
                    'cropped_indices': self.cropped_indices,
                    'origsize': self.dataOrigShape}
            file_obj = open(self.fnoutput+'.obj', 'wb')
            pickle.dump(data, file_obj, protocol=4)

        print('H5 files of images generated.')

