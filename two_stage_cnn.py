import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

class model_2input_mirrored(nn.Module):

    def __init__(self,k_dim_x=4,d_dim_x=1, k_dim_n=4,d_dim_n=1, k_dim_t=4,d_dim_t=1, in_features=1,
                 patchsize =1, labels=4, pad=3, linepad=5, zlinepad=3, f_dim = 5, params=None, channels=3):
        super(model_2input_mirrored, self).__init__()
        self.d_dim_x = d_dim_x
        self.k_dim_x = k_dim_x
        self.d_dim_n = d_dim_n
        self.k_dim_n = k_dim_n
        self.d_dim_t = d_dim_t
        self.k_dim_t = k_dim_t

        self.channels = channels
        self.params = params
        self.f_dim = f_dim
        self.pad =pad
        self.in_features = in_features
        self.orig_img_size = int(np.sqrt(in_features))
        self.img_size = patchsize #int(np.sqrt(in_features))
        self.linepad = self.img_size * (linepad * 2 + 1)
        self.labels = labels
        self.coords =3
        self.h1 = 5 *2
        self.h2 = 10 *2
        self.h2b = 5 *3

        self.prez = 3 *2

        self.lineh1 = 16 *2
        self.lineh2 = 8 *2

        self.hloc1 = 14

        self.d = patchsize
        self.totalpad = int(self.pad*self.d)
        self.neighsize = int(self.totalpad*2 + self.d)

        self.w = self.linepad
        self.zlinepad = self.img_size * (zlinepad * 2 + 1)
        self.convsize1 = int(self.linepad/2)
        self.out = int(self.w -self.convsize1) + 1
        self.convsize2 = int(self.out/2) +1
        self.lineout = int(self.out/2) +1
        self.numf1 = 10
        self.numf2 = 20
        self.lineh= 10

        # N1 input
        self.filters1 = 10 * 2 *2
        self.filters2 = 20 * 2 *2
        self.kw1 = 2
        self.kw2 = 3
        self.kw3 = 5
        self.convflatdim = self.filters2

        self.convinput = nn.Sequential(
            nn.Conv2d(self.channels, self.filters1, self.kw1, 1),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(4)
        )

        self.int_convinput = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        # Nz1 input
        self.convinputz = nn.Sequential(
            nn.Conv2d(self.channels, self.filters1, self.kw1, 1),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(4)
        )

        self.int_convinputz = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )


        # Ny1 input
        self.convinputy = nn.Sequential(
            nn.Conv2d(self.channels, self.filters1, self.kw1, 1),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(4)
        )

        self.int_convinputy = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )


        # f generation
        self.h_f1 = 20
        self.h_f2 = 30
        self.fgen = nn.Sequential(
            # nn.Linear(self.d_dim_x + self.d_dim_n + self.d_dim_t, self.h_f1),
            # nn.Linear( self.filters2*3 + self.prez, self.h_f1),
            nn.Linear(self.filters2 * 3, self.h_f1),
            nn.BatchNorm1d( self.h_f1),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f1, self.h_f2),
            nn.BatchNorm1d(self.h_f2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f2, self.f_dim)
        )


        self.h3 = 12 #* 3
        self.h4 = 6 # * 3

        # classification
        self.labelpredic = nn.Sequential(
            nn.Linear(self.f_dim, self.h3),
            nn.BatchNorm1d(self.h3),
            nn.LeakyReLU(0.1, True),
            # nn.Linear(self.h3, self.h3),
            # nn.BatchNorm1d(self.h3),
            # nn.LeakyReLU(0.1, True),
            nn.Linear(self.h3, self.h4),
            nn.BatchNorm1d(self.h4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h4, self.labels),
            nn.Softmax(dim=1)
            # nn.Tanh()
        )


    def find_nearest(self,query,target):
        Q=query.unsqueeze(1).repeat(1,target.size(0),1)
        T=target.unsqueeze(0).repeat(query.size(0),1,1)
        index=(Q-T).pow(2).sum(2).sqrt().min(1)[1]
        return target[index]

    def forward(self, X1, neigh, neigh_z, neigh_y):

        int_neighbors = self.convinput(neigh)
        feat_neigh = int_neighbors.view(int_neighbors.size(0), -1)
        neighbors = self.int_convinput(int_neighbors)
        neigh_flat = neighbors.view(neighbors.size(0), -1)

        int_neighborsz = self.convinputz(neigh_z)
        feat_neighz = int_neighborsz.view(int_neighborsz.size(0), -1)
        neighborsz = self.int_convinputz(int_neighborsz)
        neigh_flatz = neighborsz.view(neighborsz.size(0), -1)

        int_neighborsy = self.convinputy(neigh_y)
        feat_neighy = int_neighborsy.view(int_neighborsy.size(0), -1)
        neighborsy = self.int_convinputy(int_neighborsy)
        neigh_flaty = neighborsy.view(neighborsy.size(0), -1)


        feat_cat = torch.cat((neigh_flat, neigh_flatz, neigh_flaty), 1)

        f = self.fgen(feat_cat)

        label_OHE = self.labelpredic(f)

        return label_OHE, f

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad
