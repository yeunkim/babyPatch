from torch.utils.data import Dataset
import pickle
import torch
import numpy as np
import h5py

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, multiinput=False, threedim=False, coords=False, spherecoord=False, edgemap=None):
        self.multiinput = multiinput
        self.threedim = threedim
        self.coords = coords
        self.spherecoord = spherecoord
        self.edgemap = edgemap

    def __call__(self, sample):
        label, neighbors = sample['label'], sample['neighbors']

        neighbors_z, neighbors_y = sample['neighbors_z'], sample['neighbors_y']


        label = np.asarray(label).astype('int')

        if self.edgemap:
            edge = sample['edgemap']
            edge = np.asarray(edge).astype('int')
            sample.update({
                'edgemap': torch.from_numpy(edge)
            })

        try:
            neighbors = neighbors.transpose((2, 0, 1)).astype('float32')
        except:
            neighbors = np.expand_dims(neighbors, axis=2).transpose((2, 0, 1)).astype('float32')

        try:
            neighbors_z = neighbors_z.transpose((2, 0, 1)).astype('float32')
            neighbors_y = neighbors_y.transpose((2, 0, 1)).astype('float32')
        except:
            neighbors_z = np.expand_dims(neighbors_z, axis=2).transpose((2, 0, 1)).astype('float32')
            neighbors_y = np.expand_dims(neighbors_y, axis=2).transpose((2, 0, 1)).astype('float32')
        sample = {
                'label': torch.from_numpy(label),
                'neighbors': torch.from_numpy(neighbors),
                'neighbors_z': torch.from_numpy(neighbors_z),
                'neighbors_y': torch.from_numpy(neighbors_y)
                }

        return sample

class MRDataSet(Dataset):
    """MRI dataset."""

    def __init__(self, dataset, transform=None, pad=5, render=True, slices = True, channels=1, axes=(True,False,False),
                 numslices=(10,0,0)):

        self.dataset = dataset
        self.render = render
        self.transform = transform
        self.pad = pad
        self.channels = channels
        self.axes = axes
        self.numslices = np.asarray(numslices)
        self.numslices[self.numslices == 0] = 1
        self.slices = slices

        ## compute plane locations
        bounds = self.dataset.bounds
        empty_mask = np.zeros_like(self.dataset.data, dtype=np.int)
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


    def __len__(self):
        length = len(self.idxs[0])
        return length

    def __getitem__(self, idx):
        ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]

        if self.render:
            i,j, k = self.idxs[:,idx]
            neighbors = self.dataset.data[i - self.pad:i + self.pad + 1, j - self.pad:j + self.pad + 1, k]
            neighbors_z = self.dataset.data[i - self.pad:i + self.pad + 1, j, k - self.pad:k + self.pad + 1]
            neighbors_y = self.dataset.data[i, j - self.pad:j + self.pad + 1, k - self.pad:k + self.pad + 1]
            label = self.dataset.targets[i,j,k]
            indices = ravel(self.idxs[:, idx], self.shape)
        sample = {'label': label}
        sample.update({'neighbors': neighbors, 'neighbors_z': neighbors_z, 'neighbors_y': neighbors_y})

        if self.transform:
            sample = self.transform(sample)


        return sample, idx, indices #, datasetNum