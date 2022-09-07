import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

class model_2input_mirrored(nn.Module):

    def __init__(self, width=11, height=11, labels=4,f_dim = 5, params=None, channels=1,
                 spherecoords=None):
        super(model_2input_mirrored, self).__init__()

        self.channels = channels
        self.params = params
        self.f_dim = f_dim
        self.width = width
        self.height = height
        self.labels = labels
        self.coords =3
        self.spherecoords = spherecoords


        # N1 input
        self.filters1 = 48
        self.filters2 = 72
        self.filters3 = 84
        self.filters4 = 108
        def conv2d_size_out(size, kernel_size, stride=1, pad=0):
            return ((size - kernel_size + (2*pad))/ stride) + 1

        self.kw1 = int(width / 5)
        self.kw2 = int(conv2d_size_out(width, self.kw1)/ 3)
        self.kw3 = int(conv2d_size_out( conv2d_size_out(width, self.kw1), self.kw2) / 2)
        self.kw4 = int(conv2d_size_out(conv2d_size_out(conv2d_size_out(width, self.kw1), self.kw2), self.kw3) / 2)

        self.convinput = nn.Sequential(
            nn.Conv2d(self.channels, self.filters1, self.kw1),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters3, self.kw3),
            nn.BatchNorm2d(self.filters3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters3, self.filters4, self.kw4),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(self.filters4)
        )

        # Nz1 input
        self.convinputz = nn.Sequential(
            nn.Conv2d(self.channels, self.filters1, self.kw1),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters3, self.kw3),
            nn.BatchNorm2d(self.filters3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters3, self.filters4, self.kw4),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(self.filters4)
        )

        # Ny1 input
        self.convinputy = nn.Sequential(
            nn.Conv2d(self.channels, self.filters1, self.kw1),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters3, self.kw3),
            nn.BatchNorm2d(self.filters3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters3, self.filters4, self.kw4),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(self.filters4)
        )


        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.width, self.kw1), self.kw2), self.kw3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.height, self.kw1), self.kw2), self.kw3)
        linear_input_size = convw * convh * 3

        # f generation
        self.h_f1 = 40 # *2
        self.h_f2 = 50 # *2
        self.h_f3 = 60
        self.fgen = nn.Sequential(
            nn.Linear(linear_input_size , self.h_f1),
            nn.BatchNorm1d( self.h_f1),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f1, self.h_f2),
            nn.BatchNorm1d(self.h_f2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f2, self.f_dim),
            nn.BatchNorm1d(self.f_dim),
            nn.LeakyReLU(0.1, True),
        )

        self.h3 = 24 #*2
        self.h4 = 12 #*2


        # classification
        if self.spherecoords:
            self.labelpredic_spherecoord = nn.Sequential(
                nn.Linear(self.f_dim + 3, self.h3 + 2),
                nn.BatchNorm1d(self.h3 + 2),
                # nn.Dropout(0.5),
                nn.LeakyReLU(0.1, True),

                nn.Linear(self.h3 + 2, self.h3),
                nn.BatchNorm1d(self.h3),
                nn.LeakyReLU(0.1, True),

                nn.Linear(self.h3, self.h4),
                nn.BatchNorm1d(self.h4),
                # nn.Dropout(0.5),
                nn.LeakyReLU(0.1, True),
                nn.Linear(self.h4, self.labels),
                nn.Softmax(dim=1)
                # nn.Tanh()
            )
        else:
            self.labelpredic = nn.Sequential(
                nn.Linear(self.f_dim, self.h3),
                nn.BatchNorm1d(self.h3),
                nn.LeakyReLU(0.1, True),
                nn.Linear(self.h3, self.h4),
                nn.BatchNorm1d(self.h4),
                nn.LeakyReLU(0.1, True),
                nn.Linear(self.h4, self.labels),
                nn.Softmax(dim=1)
            )


    def forward(self, neigh, neigh_z, neigh_y, spherecoord=None):

        neighbors = self.convinput(neigh)
        neigh_flat = neighbors.view(neighbors.size(0), -1)

        neighborsz = self.convinputz(neigh_z)
        neigh_flatz = neighborsz.view(neighborsz.size(0), -1)

        neighborsy = self.convinputy(neigh_y.float())
        neigh_flaty = neighborsy.view(neighborsy, -1)

        feat_cat = torch.cat((neigh_flat, neigh_flatz, neigh_flaty), 1)

        f = self.fgen(feat_cat)

        if self.spherecoords:
            f_spherecoord = torch.cat((f,spherecoord.view(-1,3).float()), 1)
            label_OHE = self.labelpredic_spherecoord(f_spherecoord)
        else:
            label_OHE = self.labelpredic(f)

        _ , indices = label_OHE.max(1)

        return label_OHE, f , indices
