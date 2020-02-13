from torch.utils.data import Dataset
import pickle
import torch
import numpy as np

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, multiinput=False, threedim=False, coords=False):
        self.multiinput = multiinput
        self.threedim = threedim
        self.coords = coords

    def __call__(self, sample):
        # image1, line, zline, label, neighbors, neighbors_z, neighbors_y = sample['image1'], sample['line'], sample['zline'], \
        #                                           sample['label'], sample['neighbors'], sample['neighbors_z'], sample['neighbors_y']

        image1, label, neighbors, image2, neighbors2 = sample['image1'], sample['label'], sample['neighbors'],sample['image2'],sample['neighbors2']
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

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image1 = np.expand_dims(image1, axis=3).transpose((1, 0)).astype('float32')
        image2 = np.expand_dims(image2, axis=3).transpose((1, 0)).astype('float32')
        # line = np.expand_dims(line, axis=3).transpose((1, 0)).astype('float32')
        # zline = np.expand_dims(zline, axis=3).transpose((1, 0)).astype('float32')
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
                # neighbors_z = np.expand_dims(neighbors_z, axis=3).transpose((2, 0, 1)).astype('float32')
                # neighbors_y = np.expand_dims(neighbors_y, axis=3).transpose((2, 0, 1)).astype('float32')
                neighbors2_z = np.expand_dims(neighbors2_z, axis=3).transpose((2, 0, 1)).astype('float32')
                neighbors2_y = np.expand_dims(neighbors2_y, axis=3).transpose((2, 0, 1)).astype('float32')
        sample = {'image1': torch.from_numpy(image1),
                  'image2': torch.from_numpy(image2),
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

    def __init__(self, pkl_file, pkl_file2, transform=None, multiinput=False, miscidxs=None, threedim=False, coords=False):
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
        if miscidxs is not None:
            self.dataset.X_t1 = self.dataset.X_t1[self.miscidxs]
            self.dataset.X5 = self.dataset.X5[self.miscidxs]
            self.dataset.neighbors_t1 = self.dataset.neighbors_t1[self.miscidxs]
            self.dataset.neighbors_z_t1 = self.dataset.neighbors_z_t1[self.miscidxs]
            self.dataset.neighbors_y_t1 = self.dataset.neighbors_y_t1[self.miscidxs]

        self.transform = transform
        self.hrshape = self.dataset.dataUpsampledShape
        self.multiinput = multiinput



    def __len__(self):
        return len(self.dataset.X5)

    def __getitem__(self, idx):
        image1 = self.dataset.X[idx]
        label = self.dataset.X5[idx]
        neighbors = self.dataset.neighbors[idx]

        image2 = self.dataset2.X[idx]
        neighbors2 = self.dataset2.neighbors[idx]

        sample = {'image1': image1, 'image2': image2, 'label': label,
                  'neighbors': neighbors, 'neighbors2': neighbors2}

        if not self.threedim:
            # yline = self.dataset.line[idx]
            # zline = self.dataset.zline[idx]
            neighbors_z = self.dataset.neighbors_z[idx]
            neighbors_y = self.dataset.neighbors_y[idx]

            neighbors2_z = self.dataset2.neighbors_z[idx]
            neighbors2_y = self.dataset2.neighbors_y[idx]

            sample.update({'neighbors_z': neighbors_z, 'neighbors_y': neighbors_y,'neighbors2_z': neighbors2_z, 'neighbors2_y': neighbors2_y})
            # sample = {'image1': image1,'line': yline, 'zline': zline, 'label': label,
            #           'neighbors': neighbors, 'neighbors_z': neighbors_z, 'neighbors_y': neighbors_y}

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

        if self.transform:
            sample = self.transform(sample)

        return sample, idx, self.dataset.indices[idx]