from torch.utils.data import Dataset
import pickle
import torch
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, multiinput=False, threedim=False, coords=False, spherecoord=False):
        self.multiinput = multiinput
        self.threedim = threedim
        self.coords = coords
        self.spherecoord = spherecoord

    def __call__(self, sample):
        # image1, line, zline, label, neighbors, neighbors_z, neighbors_y = sample['image1'], sample['line'], sample['zline'], \
        #                                           sample['label'], sample['neighbors'], sample['neighbors_z'], sample['neighbors_y']

        label, neighbors, neighbors2 = sample['label'], sample['neighbors'],sample['neighbors2']
        if self.multiinput:
            image1_t1, neighbors_t1 = sample['image1_t1'], sample['neighbors_t1']

        if not self.threedim:
            neighbors_z, neighbors_y, neighbors2_z, neighbors2_y  = sample['neighbors_z'], sample['neighbors_y'],sample['neighbors2_z'], sample['neighbors2_y']

            if self.multiinput:
                neighbors_z_t1, neighbors_y_t1 = sample['neighbors_z_t1'], sample['neighbors_y_t1']
                # image1, line, zline, label, neighbors, neighbors_z, neighbors_y,\
            #     image1_t1, neighbors_t1, neighbors_z_t1, neighbors_y_t1 = sample['image1'], sample['line'], sample[
            #     'zline'], sample['label'], sample['neighbors'], sample['neighbors_z'], sample['neighbors_y'], \
            #                                                                   sample['image1_t1'], sample['neighbors_t1'], \
            #                                                             sample['neighbors_z_t1'], sample['neighbors_y_t1']

        if self.coords:
            coord = sample['coord']

        if self.spherecoord:
            spherecoord = sample['spherecoord']
            spherecoord = np.expand_dims(spherecoord, axis=1).transpose((1,0)).astype('float32')

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image1 = np.expand_dims(image1, axis=3).transpose((1, 0)).astype('float32')
        # image2 = np.expand_dims(image2, axis=3).transpose((1, 0)).astype('float32')
        # line = np.expand_dims(line, axis=3).transpose((1, 0)).astype('float32')
        # zline = np.expand_dims(zline, axis=3).transpose((1, 0)).astype('float32')
        label = np.asarray(label).astype('int')
        if self.threedim:
            neighbors = np.expand_dims(neighbors, axis=4).transpose((3, 0, 1, 2)).astype('float32')
        else:
            try:
                neighbors = np.expand_dims(neighbors, axis=2).transpose((2, 0, 1)).astype('float32')
                neighbors2 = neighbors2.transpose((2, 0, 1)).astype('float32')
            except:
                neighbors = neighbors.transpose((2, 0, 1)).astype('float32')
                neighbors2 = neighbors2.transpose((2, 0, 1)).astype('float32')
        if not self.threedim:
            try:
                neighbors_z = np.expand_dims(neighbors_z, axis=2).transpose((2, 0, 1)).astype('float32')
                neighbors_y = np.expand_dims(neighbors_y, axis=2).transpose((2, 0, 1)).astype('float32')
                neighbors2_z = neighbors2_z.transpose((2, 0, 1)).astype('float32')
                neighbors2_y = neighbors2_y.transpose((2, 0, 1)).astype('float32')
            except:
                neighbors_z = neighbors_z.transpose((2, 0, 1)).astype('float32')
                neighbors_y = neighbors_y.transpose((2, 0, 1)).astype('float32')
                neighbors2_z = neighbors2_z.transpose((2, 0, 1)).astype('float32')
                neighbors2_y = neighbors2_y.transpose((2, 0, 1)).astype('float32')
        sample = { #'image1': torch.from_numpy(image1),
                  #'image2': torch.from_numpy(image2),
                # 'line': torch.from_numpy(line),
                # 'zline': torch.from_numpy(zline),
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

        if self.spherecoord:
            sample.update({
                'spherecoord': torch.from_numpy(spherecoord)
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

    def __init__(self, pkl_file, pkl_file2, transform=None, multiinput=False, miscidxs=None, threedim=False, coords=False,
                 pad=6, spherecoord=False):
        """
        Args:
            pickle_file (string): Path to the pickle file with annotations.
            transform
        """
        file_obj = open(pkl_file, 'rb')
        file_obj2 = open(pkl_file2, 'rb')
        self.dataset = pickle.load(file_obj)
        self.dataset2 = pickle.load(file_obj2)
        self.miscidxs = miscidxs
        self.threedim = threedim
        self.coords = coords
        self.spherecoord = spherecoord
        if miscidxs is not None:
            self.dataset.X_t1 = self.dataset.X_t1[self.miscidxs]
            self.dataset.X5 = self.dataset.X5[self.miscidxs]
            self.dataset.neighbors_t1 = self.dataset.neighbors_t1[self.miscidxs]
            self.dataset.neighbors_z_t1 = self.dataset.neighbors_z_t1[self.miscidxs]
            self.dataset.neighbors_y_t1 = self.dataset.neighbors_y_t1[self.miscidxs]

        self.transform = transform
        self.hrshape = self.dataset.dataUpsampledShape
        self.multiinput = multiinput
        self.pad = pad



    def __len__(self):
        return len(self.dataset.X5)

    def __getitem__(self, idx):
        label = self.dataset.X5[idx]
        neighbors = self.dataset.neighbors[idx]

        neighbors2 = self.dataset2.neighbors[idx]

        sample = {'label': label,
                  'neighbors': neighbors, 'neighbors2': neighbors2}

        if not self.threedim:
            neighbors_z = self.dataset.neighbors_z[idx]
            neighbors_y = self.dataset.neighbors_y[idx]

            neighbors2_z = self.dataset2.neighbors_z[idx]
            neighbors2_y = self.dataset2.neighbors_y[idx]

            sample.update({'neighbors_z': neighbors_z, 'neighbors_y': neighbors_y,'neighbors2_z': neighbors2_z,
                           'neighbors2_y': neighbors2_y})

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

        # self.indices_neighx = []
        # self.indices_neighy = []
        # self.indices_neighz = []
        #
        # width = (1 + (2 * self.pad))
        # size = self.dataset.dataOrigShape[:3]
        # i, j, k = np.unravel_index(self.dataset.indices[idx], size)
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

        return sample, idx, self.dataset.indices[idx], self.dataset.dataNum
               # self.indices_neighx, self.indices_neighz,self.indices_neighy