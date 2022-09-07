from torch.utils.data import Dataset
import pickle
import torch
import numpy as np
import h5py

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, multiinput=False, threedim=False, coords=False, spherecoord=False):
        self.multiinput = multiinput
        self.threedim = threedim
        self.coords = coords
        self.spherecoord = spherecoord

    def __call__(self, sample):
        neigh1, neigh2, uncert1, uncert2 = sample['neighbors1'], sample['neighbors2'], \
                                                  sample['uncert1'], sample['uncert2']
        # label = np.asarray(label).astype('int')
        try:
            neigh1 = neigh1.transpose((2, 0, 1)).astype('float32')
        except:
            neigh1 = np.expand_dims(neigh1, axis=2).transpose((2, 0, 1)).astype('float32')
        # TODO: convert to tensor
        try:
            neigh2 = neigh2.transpose((2, 0, 1)).astype('float32')
        except:
            neigh2 = np.expand_dims(neigh2, axis=2).transpose((2, 0, 1)).astype('float32')

        try:
            uncert1 = uncert1.transpose((2, 0, 1)).astype('float32')
            uncert2 = uncert2.transpose((2, 0, 1)).astype('float32')
        except:
            uncert1 = np.expand_dims(uncert1, axis=2).transpose((2, 0, 1)).astype('float32')
            uncert2 = np.expand_dims(uncert2, axis=2).transpose((2, 0, 1)).astype('float32')
        sample = {
                # 'label': torch.from_numpy(label),
                'neighbors1': torch.from_numpy(neigh1),
                'neighbors2': torch.from_numpy(neigh2),
                'uncert1': torch.from_numpy(uncert1),
                'uncert2': torch.from_numpy(uncert2)
                }

        return sample

class MRDataSet(Dataset):
    """MRI dataset."""

    def __init__(self, dataset, dataset2, transform=None, pad=5, render=True, pkl = False, slices = True, channels=1, axes=(True,False,False),
                 numslices=(10,0,0), planes=(True, False, True)):
        """
        Args:
            pickle_file (string): Path to the pickle file with annotations.
            transform
        """
        self.dataset = dataset
        self.dataset2 = dataset2
        self.render = render
        self.transform = transform
        self.pad = pad
        self.channels = channels
        self.axes = axes
        self.numslices = np.asarray(numslices)
        self.numslices[self.numslices == 0] = 1
        self.slices = slices
        self.planes = planes
        self.plane1, self.plane2 = np.where(np.asarray(planes) == True)[0]

        ## compute plane locations
        bounds = self.dataset.bounds
        shape = self.dataset.data.shape
        empty_mask = np.zeros(shape[:3], dtype=np.int)
        self.shape = empty_mask.shape
        if self.slices:
            dims = np.asarray([bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0], bounds[2][1] - bounds[2][0]])
            intervals = (dims/np.asarray(self.numslices)).astype(np.int)
            self.axis_locs = [np.arange(bounds[i][0],bounds[i][1], intervals[i]) for i in range(0,len(self.axes))]
            # select slices from both data and mask, and label if label is not None
            if self.axes[0]:
                empty_mask[self.axis_locs[0],:,:] = 1
            if self.axes[1]:
                empty_mask[:,self.axis_locs[1],:] = 1
            if self.axes[2]:
                empty_mask[:, :, self.axis_locs[2]] = 1
        else:
            empty_mask[self.dataset.mask[:] > 0] = 1
        if not np.all(self.dataset.targets == None):
            empty_mask[(self.dataset.mask[:] == 0) | (self.dataset.targets[:] == 4)]= 0
        else:
            empty_mask[(self.dataset.mask[:] == 0)] = 0
        self.idxs = np.asarray(np.where(empty_mask>0))
        # print('stop')
        # print out json files with details using logging_functions
    def planes_to_patches(self, dataset, plane1, plane2, coords):
        i,j,k = coords
        finder = {
            0: dataset.data[i - self.pad:i + self.pad + 1, j - self.pad:j + self.pad + 1, k],
            1: dataset.data[i - self.pad:i + self.pad + 1, j, k - self.pad:k + self.pad + 1],
            2: dataset.data[i, j - self.pad:j + self.pad + 1, k - self.pad:k + self.pad + 1]
        }
        neigh1 = finder.get(plane1)
        neigh2 = finder.get(plane2)
        return neigh1, neigh2

    def __len__(self):
        length = len(self.idxs[0])
        return length

    def __getitem__(self, idx):
        ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]
        i,j, k = self.idxs[:,idx]
        neigh1, neigh2 = self.planes_to_patches(self.dataset, self.plane1, self.plane2, (i,j,k))
        indices = ravel(self.idxs[:,idx],self.shape)
        # datasetNum = self.dataset.attrs['datasetNum']
        # label = self.dataset.targets[i,j,k]

        uncert1, uncert2 = self.planes_to_patches(self.dataset2, self.plane1, self.plane2, (i, j, k))
        # sample = {'label': label}
        sample = ({'neighbors1': neigh1, 'neighbors2': neigh2,
                       'uncert1': uncert1, 'uncert2': uncert2})

        if self.transform:
            sample = self.transform(sample)


        return sample, idx, indices # datasetNum