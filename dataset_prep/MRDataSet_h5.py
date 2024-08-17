from torch.utils.data import Dataset
import pickle
import torch
import numpy as np
import h5py

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, multiinput=False, threedim=False, coords=False):
        self.multiinput = multiinput
        self.threedim = threedim
        self.coords = coords

    def __call__(self, sample):

        label, neighbors = sample['label'], sample['neighbors']
        if self.multiinput:
            image1_t1, neighbors_t1 = sample['image1_t1'], sample['neighbors_t1']

        if not self.threedim:
            neighbors_z, neighbors_y = sample['neighbors_z'], sample['neighbors_y']

            if self.multiinput:
                neighbors_z_t1, neighbors_y_t1 = sample['neighbors_z_t1'], sample['neighbors_y_t1']

        if self.coords:
            coord = sample['coord']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        label = np.asarray(label).astype('int')
        if self.threedim:
            neighbors = np.expand_dims(neighbors, axis=4).transpose((3, 0, 1, 2)).astype('float32')
        else:
            try:
                neighbors = neighbors.transpose((2, 0, 1)).astype('float32')
            except:
                neighbors = np.expand_dims(neighbors, axis=3).transpose((2, 0, 1)).astype('float32')
        if not self.threedim:
            try:
                neighbors_z = neighbors_z.transpose((2, 0, 1)).astype('float32')
                neighbors_y = neighbors_y.transpose((2, 0, 1)).astype('float32')
            except:
                neighbors_z = np.expand_dims(neighbors_z, axis=3).transpose((2, 0, 1)).astype('float32')
                neighbors_y = np.expand_dims(neighbors_y, axis=3).transpose((2, 0, 1)).astype('float32')
        sample = {
                # 'line': torch.from_numpy(line),
                # 'zline': torch.from_numpy(zline),
                'label': torch.from_numpy(label),
                'neighbors': torch.from_numpy(neighbors),
                }
        if not self.threedim:
            sample.update({
                'neighbors_z': torch.from_numpy(neighbors_z),
                'neighbors_y': torch.from_numpy(neighbors_y)
            })

        if self.multiinput:

            if self.threedim:
                neighbors_t1 = np.expand_dims(neighbors_t1, axis=4).transpose((3, 0, 1, 2)).astype('float32')
            else:
                try:
                    neighbors_t1 = neighbors_t1.transpose((2, 0, 1)).astype('float32')
                except:
                    neighbors_t1 = np.expand_dims(neighbors_t1, axis=3).transpose((2, 0, 1)).astype('float32')
            sample.update({'image1_t1': torch.from_numpy(image1_t1),
                           'neighbors_t1': torch.from_numpy(neighbors_t1),

                           })
            if not self.threedim:
                try:
                    neighbors_z_t1 = neighbors_z_t1.transpose((2, 0, 1)).astype('float32')
                    neighbors_y_t1 = neighbors_y_t1.transpose((2, 0, 1)).astype('float32')
                except:
                    neighbors_z_t1 = np.expand_dims(neighbors_z_t1, axis=3).transpose((2, 0, 1)).astype('float32')
                    neighbors_y_t1 = np.expand_dims(neighbors_y_t1, axis=3).transpose((2, 0, 1)).astype('float32')
                sample.update({
                    'neighbors_z_t1': torch.from_numpy(neighbors_z_t1),
                    'neighbors_y_t1': torch.from_numpy(neighbors_y_t1)
                })

        if self.coords:
            coord = np.asarray(coord).astype('int')
            sample.update({'coord': torch.from_numpy(coord)})

        return sample

class ToTensor_h5(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, x):
        self.x = x

    def __call__(self):

        try:
            self.x = self.x.transpose((2, 0, 1)).astype('float32')
        except:
            self.x = np.expand_dims(self.x, axis=3).transpose((2, 0, 1)).astype('float32')

        return self.x

class MRDataSet(Dataset):
    """MRI dataset."""

    def __init__(self, h5file, pad=5, transform=None, multiinput=False, pkl=None,
                 miscidxs=None, threedim=False, coords=False):
        """
        Args:
            h5file (string): Path to the pickle file with annotations.
            transform
        """

        #with h5py.File(h5file, 'r') as f:
        self.h5file = h5file
        self.pkl = pkl
        self.pad = pad
        self.miscidxs = miscidxs
        self.threedim = threedim
        self.coords = coords
        if miscidxs is not None:
            self.dataset.X_t1 = self.dataset.X_t1[self.miscidxs]
            self.dataset.X5 = self.dataset.X5[self.miscidxs]
            self.dataset.neighbors_t1 = self.dataset.neighbors_t1[self.miscidxs]
            self.dataset.neighbors_z_t1 = self.dataset.neighbors_z_t1[self.miscidxs]
            self.dataset.neighbors_y_t1 = self.dataset.neighbors_y_t1[self.miscidxs]

        self.transform = transform
        self.multiinput = multiinput



    def __len__(self):
        # return len(self.dataset['targets'])
        if self.pkl is None:
            with h5py.File(self.h5file, 'r') as f:
                tmp = f
                return len(tmp['indices'])
        else:
            file_obj = open(self.pkl, 'rb')
            tmp = pickle.load(file_obj)
            return len(tmp['indices'])

    def __getitem__(self, idx):

        if self.pkl is None:
            with h5py.File(self.h5file, 'r') as f:
                self.dataset = f
                i, j, k = np.unravel_index(self.dataset['cropped_indices'][idx], self.dataset['data'].shape)

                label = self.dataset['targets'][i,j,k]
                neighbors = self.dataset['data'][i - self.pad:i + self.pad + 1, j - self.pad:j + self.pad + 1, k]
                neighbors_z = self.dataset['data'][i - self.pad:i + self.pad+1, j, k - self.pad:k + self.pad + 1]
                neighbors_y = self.dataset['data'][i , j - self.pad:j + self.pad + 1, k - self.pad:k + self.pad + 1]
                indices = self.dataset['indices'][idx]

                sample = {'label': label,
                          'neighbors': neighbors, 'neighbors_y': neighbors_y, 'neighbors_z': neighbors_z}
        else:
            file_obj = open(self.pkl, 'rb')
            self.dataset = pickle.load(file_obj)
            i, j, k = np.unravel_index(self.dataset['cropped_indices'][idx], self.dataset['data'].shape)
            label = self.dataset['targets'][i, j, k]
            neighbors = self.dataset['data'][i - self.pad:i + self.pad + 1, j - self.pad:j + self.pad + 1, k]
            neighbors_z = self.dataset['data'][i - self.pad:i + self.pad + 1, j, k - self.pad:k + self.pad + 1]
            neighbors_y = self.dataset['data'][i, j - self.pad:j + self.pad + 1, k - self.pad:k + self.pad + 1]
            indices = self.dataset['indices'][idx]

            sample = {'label': label,
                      'neighbors': neighbors, 'neighbors_y': neighbors_y, 'neighbors_z': neighbors_z}
        # label = self.dataset['targets'][idx]
        # neighbors = self.dataset['neighbors'][idx]
        #
        # sample = {'label': label,
        #           'neighbors': neighbors}
        #
        # if not self.threedim:
        #
        #     neighbors_z = self.dataset['neighbors_z'][idx]
        #     neighbors_y = self.dataset['neighbors_y'][idx]
        #
        #     sample.update({'neighbors_z': neighbors_z, 'neighbors_y': neighbors_y})
        #     # sample = {'image1': image1,'line': yline, 'zline': zline, 'label': label,
        #     #           'neighbors': neighbors, 'neighbors_z': neighbors_z, 'neighbors_y': neighbors_y}
        #
        # if self.multiinput:
        #     neighbors_t1 = self.dataset.neighbors_t1[idx]
        #
        #     sample.update({'neighbors_t1': neighbors_t1})
        #     if not self.threedim:
        #         neighbors_z_t1 = self.dataset.neighbors_z_t1[idx]
        #         neighbors_y_t1 = self.dataset.neighbors_y_t1[idx]
        #         sample.update({'neighbors_z_t1': neighbors_z_t1,
        #                        'neighbors_y_t1': neighbors_y_t1})
        #
        # if self.coords:
        #     coord = self.dataset.coordsvec[idx]
        #     sample.update({'coord': coord})

        if self.transform:
            sample = self.transform(sample)

        return sample, idx, indices