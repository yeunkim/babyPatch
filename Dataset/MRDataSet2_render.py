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

        if self.coords:
            coord = sample['coord']

        if self.spherecoord:
            spherecoord = sample['spherecoord']
            spherecoord = np.expand_dims(spherecoord, axis=1).transpose((1,0)).astype('float32')

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
                'neighbors': torch.from_numpy(neighbors)
                }
        if not self.threedim:
            sample.update({
                'neighbors_z': torch.from_numpy(neighbors_z),
                'neighbors_y': torch.from_numpy(neighbors_y)
            })

        if self.spherecoord:
            sample.update({
                'spherecoord': torch.from_numpy(spherecoord)
            })

        if self.coords:
            coord = np.asarray(coord).astype('int')
            sample.update({'coord': torch.from_numpy(coord)})

        return sample

class MRDataSet(Dataset):
    """MRI dataset."""

    def __init__(self, file, transform=None, multiinput=False, miscidxs=None, threedim=False, coords=False,
                 pad=5, spherecoord=False, render=False, pkl = True, channels=1, edgemap=None):
        """
        Args:
            pickle_file (string): Path to the pickle file with annotations.
            transform
        """
        self.pkl = pkl
        self.file = file
        if self.pkl:
            file_obj = open(file, 'rb')
            self.dataset = pickle.load(file_obj)
        self.miscidxs = miscidxs
        self.threedim = threedim
        self.coords = coords
        self.spherecoord = spherecoord
        self.render = render
        # if miscidxs is not None:
        #     self.dataset.X_t1 = self.dataset.X_t1[self.miscidxs]
        #     self.dataset.X5 = self.dataset.X5[self.miscidxs]
        #     self.dataset.neighbors_t1 = self.dataset.neighbors_t1[self.miscidxs]
        #     self.dataset.neighbors_z_t1 = self.dataset.neighbors_z_t1[self.miscidxs]
        #     self.dataset.neighbors_y_t1 = self.dataset.neighbors_y_t1[self.miscidxs]

        self.transform = transform
        self.multiinput = multiinput
        self.pad = pad
        self.channels = channels
        self.edgemap = edgemap


    def __len__(self):
        length = len(self.dataset['cropped_indices'])
        return length

    def __getitem__(self, idx):

        if self.pkl:
            if self.render:
                i, j, k = np.unravel_index(self.dataset['cropped_indices'][idx], self.dataset['cropped_size'])
                neighbors = self.dataset['data'][i - self.pad:i + self.pad + 1, j - self.pad:j + self.pad + 1, k]
                neighbors_z = self.dataset['data'][i - self.pad:i + self.pad + 1, j, k - self.pad:k + self.pad + 1]
                neighbors_y = self.dataset['data'][i, j - self.pad:j + self.pad + 1, k - self.pad:k + self.pad + 1]
                indices = self.dataset['indices'][idx]
                datasetNum = self.dataset['datasetNum']
                cropped_indices = self.dataset['cropped_indices'][idx]
                label = self.dataset['targets'][i,j,k]
                if self.edgemap:
                    edge = self.dataset['edgemap'][i,j,k]
                # sample_id = self.dataset['cropped_indices_map'][cropped_indices]
            else:
                label = self.dataset.X5[idx]
                neighbors = self.dataset.neighbors[idx]
                neighbors_z = self.dataset.neighbors_z[idx]
                neighbors_y = self.dataset.neighbors_y[idx]
                indices = self.dataset.indices[idx]
                cropped_indices = self.dataset.cropped_indices[idx]
            sample = {'label': label}
            sample.update({'neighbors': neighbors, 'neighbors_z': neighbors_z, 'neighbors_y': neighbors_y})
        else:
            if self.render:
                with h5py.File(self.file, 'r') as self.dataset:
                    i, j, k = np.unravel_index(self.dataset['cropped_indices'][idx], self.dataset['cropped_size'])
                    neighbors = self.dataset['data'][i - self.pad:i + self.pad + 1, j - self.pad:j + self.pad + 1, k]
                    neighbors_z = self.dataset['data'][i - self.pad:i + self.pad + 1, j, k - self.pad:k + self.pad + 1]
                    neighbors_y = self.dataset['data'][i, j - self.pad:j + self.pad + 1, k - self.pad:k + self.pad + 1]
                    indices = self.dataset['indices'][idx]
                    datasetNum = self.dataset['datasetNum']
                    cropped_indices = self.dataset['cropped_indices'][idx]
                    label = self.dataset['targets'][i,j,k]
            # else:
            #     with h5py.File(self.file, 'r') as self.dataset:
            #         label = self.dataset.X5[idx]
            #         neighbors = self.dataset.neighbors[idx]
            #         neighbors_z = self.dataset.neighbors_z[idx]
            #         neighbors_y = self.dataset.neighbors_y[idx]
            #         indices = self.dataset.indices[idx]
            #         cropped_indices = self.dataset.cropped_indices[idx]
            sample = {'label': label}
            sample.update({'neighbors': neighbors, 'neighbors_z': neighbors_z, 'neighbors_y': neighbors_y})
        if self.multiinput:
            image1_t1 = self.dataset.X_t1[idx]
            neighbors_t1 = self.dataset.neighbors_t1[idx]

            sample.update({'image1_t1': image1_t1, 'neighbors_t1': neighbors_t1})
            if not self.threedim:
                neighbors_z_t1 = self.dataset.neighbors_z_t1[idx]
                neighbors_y_t1 = self.dataset.neighbors_y_t1[idx]
                sample.update({'neighbors_z_t1': neighbors_z_t1,
                               'neighbors_y_t1': neighbors_y_t1})

        if self.coords:
            coord = self.dataset.coordsvec[idx]
            sample.update({'coord': coord})

        if self.spherecoord:
            sample.update({'spherecoord': self.dataset.sphericalcoordinates[idx]})

        if self.transform:
            sample = self.transform(sample)

        # self.indices_neighx =[]
        # self.indices_neighy = []
        # self.indices_neighz = []
        #
        # width = (1 + (2 * self.pad))
        # size = self.dataset.dataOrigShape[:3]
        # i,j,k = np.unravel_index(self.dataset.indices[idx], size)
        # # for num in len(i):
        # padslicex, padslicey = np.meshgrid(range(i - self.pad, i + self.pad + 1), range(j - self.pad, j + self.pad + 1))
        # padslicezc = np.array([[k] * width] * width)
        # # reshape
        # padslicex = np.expand_dims(padslicex, axis=0)
        # padsliceyT = np.expand_dims(np.transpose(padslicey), axis=0)
        # padslicey = np.expand_dims(padslicey, axis=0)
        # padslicezc = np.expand_dims(padslicezc, axis=0)
        # self.indices_neighx.append(np.concatenate([padslicex, padslicey, padslicezc]))
        # _, padslicez = np.meshgrid(range(i - self.pad, i + self.pad + 1), range(k - self.pad, k + self.pad + 1))
        # padsliceyc = np.array([[j] * width] * width)
        # padslicexc = np.array([[i] * width] * width)
        # # reshape
        # padslicez = np.expand_dims(padslicez, axis=0)
        # padsliceyc = np.expand_dims(padsliceyc, axis=0)
        # padslicexc = np.expand_dims(padslicexc, axis=0)
        # self.indices_neighz.append(np.concatenate([padslicex, padsliceyc, padslicez]))
        # self.indices_neighy.append(np.concatenate([padslicexc, padsliceyT, padslicez]))
        #
        # self.indices_neighx = np.array(self.indices_neighx)
        # self.indices_neighy = np.array(self.indices_neighy)
        # self.indices_neighz = np.array(self.indices_neighz)

        return sample, idx, indices, cropped_indices, datasetNum
               # self.indices_neighx, self.indices_neighz,self.indices_neighy