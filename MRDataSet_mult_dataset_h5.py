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
        label, neighbors, neighbors2 = sample['label'], sample['neighbors'],sample['neighbors2']
        if self.multiinput:
            image1_t1, neighbors_t1 = sample['image1_t1'], sample['neighbors_t1']

        if not self.threedim:
            neighbors_z, neighbors_y, neighbors2_z, neighbors2_y  = sample['neighbors_z'], sample['neighbors_y'],sample['neighbors2_z'], sample['neighbors2_y']

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
                neighbors2 = neighbors2.transpose((2, 0, 1)).astype('float32')
            except:
                # neighbors = np.expand_dims(neighbors, axis=3).transpose((2, 0, 1)).astype('float32')
                neighbors2 = np.expand_dims(neighbors2, axis=3).transpose((2, 0, 1)).astype('float32')
        if not self.threedim:
            try:
                neighbors_z = neighbors_z.transpose((2, 0, 1)).astype('float32')
                neighbors_y = neighbors_y.transpose((2, 0, 1)).astype('float32')
                neighbors2_z = neighbors2_z.transpose((2, 0, 1)).astype('float32')
                neighbors2_y = neighbors2_y.transpose((2, 0, 1)).astype('float32')
            except:
                neighbors2_z = np.expand_dims(neighbors2_z, axis=3).transpose((2, 0, 1)).astype('float32')
                neighbors2_y = np.expand_dims(neighbors2_y, axis=3).transpose((2, 0, 1)).astype('float32')
        sample = {
                'label': torch.from_numpy(label),
                'neighbors': torch.from_numpy(neighbors),
                'neighbors2': torch.from_numpy(neighbors2),

                }
        if not self.threedim:
            sample.update({
                'neighbors_z': torch.from_numpy(neighbors_z),
                'neighbors_y': torch.from_numpy(neighbors_y),
                'neighbors2_z': torch.from_numpy(neighbors2_z),
                'neighbors2_y': torch.from_numpy(neighbors2_y)
            })

        if self.multiinput:
            image1_t1 = np.expand_dims(image1_t1, axis=3).transpose((1, 0)).astype('float32')

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

class MRDataSet(Dataset):
    """MRI dataset."""

    def __init__(self, h5file, h5file2, pad=5,
                 transform=None, multiinput=False, miscidxs=None, threedim=False, coords=False):
        """
        Args:
            pickle_file (string): Path to the pickle file with annotations.
            transform
        """
        self.h5file = h5file
        self.h5file2 = h5file2
        self.pad = pad
        self.miscidxs = miscidxs
        self.threedim = threedim
        self.coords = coords
        # if miscidxs is not None:
        #     self.dataset.X5 = self.dataset.X5[self.miscidxs]
        #     self.dataset.neighbors_t1 = self.dataset.neighbors_t1[self.miscidxs]
        #     self.dataset.neighbors_z_t1 = self.dataset.neighbors_z_t1[self.miscidxs]
        #     self.dataset.neighbors_y_t1 = self.dataset.neighbors_y_t1[self.miscidxs]

        self.transform = transform
        self.multiinput = multiinput



    def __len__(self):
        with h5py.File(self.h5file, 'r') as f:
            self.dataset = f
            # return len(self.dataset['targets'])
            return len(self.dataset['indices'])

    def __getitem__(self, idx):

        with h5py.File(self.h5file, 'r') as f:
            self.dataset = f
            i, j, k = np.unravel_index(self.dataset['cropped_indices'][idx], self.dataset['data'].shape[:3])

            label = self.dataset['targets'][i, j, k]
            neighbors = self.dataset['data'][i - self.pad:i + self.pad + 1, j - self.pad:j + self.pad + 1, k]
            neighbors_z = self.dataset['data'][i - self.pad:i + self.pad + 1, j, k - self.pad:k + self.pad + 1]
            neighbors_y = self.dataset['data'][i, j - self.pad:j + self.pad + 1, k - self.pad:k + self.pad + 1]
            indices = self.dataset['indices'][idx]

            sample = {'label': label,
                      'neighbors': neighbors, 'neighbors_y': neighbors_y, 'neighbors_z': neighbors_z}

        with h5py.File(self.h5file2, 'r') as f:

        # if not self.threedim:
            self.dataset = f
            neighbors2 = self.dataset['data'][i - self.pad:i + self.pad + 1, j - self.pad:j + self.pad + 1, k]
            neighbors2_z = self.dataset['data'][i - self.pad:i + self.pad + 1, j, k - self.pad:k + self.pad + 1]
            neighbors2_y = self.dataset['data'][i, j - self.pad:j + self.pad + 1, k - self.pad:k + self.pad + 1]

            sample.update({'neighbors2': neighbors2,
                           'neighbors2_z': neighbors2_z,
                           'neighbors2_y': neighbors2_y})

            # if self.multiinput:
            #     image1_t1 = self.dataset.X_t1[idx]
            #     neighbors_t1 = self.dataset.neighbors_t1[idx]
            #
            #     sample.update({'image1_t1': image1_t1, 'neighbors_t1': neighbors_t1})
            #     if not self.threedim:
            #         neighbors_z_t1 = self.dataset.neighbors_z_t1[idx]
            #         neighbors_y_t1 = self.dataset.neighbors_y_t1[idx]
            #         sample.update({'neighbors_z_t1': neighbors_z_t1,
            #                        'neighbors_y_t1': neighbors_y_t1})
            #
            # if self.coords:
            #     coord = self.dataset.coordsvec[idx]
            #     sample.update({'coord': coord})
            #
        if self.transform:
            sample = self.transform(sample)

        return sample, idx, indices