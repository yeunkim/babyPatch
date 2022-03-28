
from torch.utils.data import Dataset
import torch
import numpy as np
from skimage.draw import disk

class shapes(object):
    def __init__(self):
        self.square = np.asarray([[0, 0, 0, 0, 0],
                                [0, 1, 1, 1, 0],
                                [0, 1, 1, 1, 0],
                                [0, 1, 1, 1, 0],
                                [0, 0, 0, 0, 0]], dtype=np.uint8)
        self.triangle = np.asarray([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0]], dtype=np.uint8)
        # self.circle = np.asarray([
        #     [0, 0, 1, 1, 0, 0],
        #     [0, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 0],
        #     [0, 1, 1, 1, 1, 0],
        #     [0, 0, 1, 1, 0, 0]], dtype=np.uint8)
        self.diamond = np.asarray([[0, 0, 1, 0, 0],
                                [0, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 0],
                                [0, 0, 1, 0, 0]], dtype=np.uint8)
        self.circle = np.zeros((11,11), dtype=np.uint8)
        rr, cc = disk((5, 5), 4)
        self.circle[rr,cc] = 1
        self.circle = self.circle.reshape((1,1,11,11))
        self.aff_matrices = np.asarray([

            [[0.5, 0, 0],
             [0, 0.5, 0],
             [0, 0, 1]],

            [[1.5, 0, 0],
             [0, 1.5, 0],
             [0, 0, 1]],

            [[1, 0, 1],
             [0, 1, 0],
             [0, 0, 1]],

            [[1, 0, 0],
             [0, 1, 1],
             [0, 0, 1]],

            [[1, 0, -1],
             [0, 1, 0],
             [0, 0, 1]],

            [[1, 0, 0],
             [0, 1, -1],
             [0, 0, 1]],

            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]

            # [[1, 0, 0],
            #  [0.5, 1, 0],
            #  [0, 0, 1]],
            #
            # [[1, 0.5, 0],
            #  [0, 1, 0],
            #  [0, 0, 1]],

            # [[1, 0, 0],
            #  [0, 1, 0],
            #  [0, 0, 1]]
        ])

class ToTensor(object):
    ''' convert nd arrays to tensors'''

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = np.expand_dims(image, 2)
        label = np.expand_dims(label, 2)
        image = image.transpose((2,0,1)).astype('float32')
        label = label.transpose((2, 0, 1)).astype('float32')

        sample = {
            'image' : torch.from_numpy(image),
            'label' : torch.from_numpy(label)
        }

        return sample

class testDataset(Dataset):
    def __init__(self, image, label, transform):
        self.image = image
        self.label = label
        self.transform = transform

    def __len__(self):
        return self.image.shape[0]

    def __getitem__(self, idx):
        sample = {
            'image' : self.image[idx],
            'label' : self.label[idx]
        }
        sample = self.transform(sample)

        return sample, idx
