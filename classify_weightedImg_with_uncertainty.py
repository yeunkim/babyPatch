import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

class model(nn.Module):

    def __init__(self, f1, f2, f3, kw1, kw2, kw3, pad=6, f_dim = 4, channels=3,
                 channels2 = 3):
        super(model, self).__init__()

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        self.kw1 = kw1
        self.kw2 = kw2
        self.kw3 = kw3

        self.channels = channels
        self.f_dim = f_dim
        self.pad = pad

        self.filters1 = 10 * 2 * 2  # *2
        self.filters2 = 20 * 2 * 2

        self.channels2 = channels2
        self.width = (2*self.pad)+1

        self.convinput_mod = nn.Sequential(
            nn.Conv2d(1, self.f1, self.kw1, 1, 0),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2, 1, 1),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f3, self.kw3, 1, 2),
            nn.BatchNorm2d(self.f3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f3, 1, self.kw1, 1, 0)
        )

        self.convinputz_mod = nn.Sequential(
            nn.Conv2d(1, self.f1, self.kw1, 1, 0),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2, 1, 1),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f3, self.kw3, 1, 2),
            nn.BatchNorm2d(self.f3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f3, 1, self.kw1, 1, 0)
        )

        self.convinputy_mod = nn.Sequential(
            nn.Conv2d(1, self.f1, self.kw1, 1, 0),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2, 1, 1),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f3, self.kw3, 1, 2),
            nn.BatchNorm2d(self.f3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f3, 1, self.kw1, 1, 0)
        )

        #TODO: change the convolutions to discrete VAE? :)

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
            nn.MaxPool2d(4),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        self.convinput_uncert = nn.Sequential(
            nn.Conv2d(self.channels2, self.filters1, self.kw1, 1),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        ## combine uncertainty with feature images
        self.h_uncert = 10  # *2
        self.h_uncert2 = 20  # *2
        self.uncert = nn.Sequential(
            nn.Linear(self.filters2 * 2, self.h_uncert),
            nn.BatchNorm1d(self.h_uncert),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert, self.h_uncert2),
            nn.BatchNorm1d(self.h_uncert2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert2, self.filters2),
            nn.BatchNorm1d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.filters2, self.width ** 2),
            nn.BatchNorm1d(self.width ** 2),
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
            nn.MaxPool2d(4),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        self.convinputz_uncert = nn.Sequential(
            nn.Conv2d(self.channels2, self.filters1, self.kw1, 1),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        self.uncertz = nn.Sequential(
            nn.Linear(self.filters2 * 2, self.h_uncert),
            nn.BatchNorm1d(self.h_uncert),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert, self.h_uncert2),
            nn.BatchNorm1d(self.h_uncert2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert2, self.filters2),
            nn.BatchNorm1d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.filters2, self.width ** 2),
            nn.BatchNorm1d(self.width ** 2),
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
            nn.MaxPool2d(4),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        self.convinputy_uncert = nn.Sequential(
            nn.Conv2d(self.channels2, self.filters1, self.kw1, 1),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(4),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        self.uncerty = nn.Sequential(
            nn.Linear(self.filters2 * 2, self.h_uncert),
            nn.BatchNorm1d(self.h_uncert),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert, self.h_uncert2),
            nn.BatchNorm1d(self.h_uncert2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert2, self.filters2),
            nn.BatchNorm1d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.filters2, self.width**2),
            nn.BatchNorm1d(self.width**2),
            nn.LeakyReLU(0.1, True)
        )

        # f generation
        self.h_f1 = 20  # *2
        self.h_f2 = 30  # *2
        self.fgen = nn.Sequential(
            nn.Linear(self.filters2 * 3, self.h_f1),
            nn.BatchNorm1d(self.h_f1),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f1, self.h_f2),
            nn.BatchNorm1d(self.h_f2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f2, self.f_dim)
        )

    def forward(self, neigh, neigh_z, neigh_y, uncert, uncert_z, uncert_y):

        ################################################################
        ################ cycle 1 ################

        ################################################################
        ################ feature extraction ################

        neighbors = self.convinput(neigh)
        neigh_flat = neighbors.view(neighbors.size(0), -1)

        ## compute with uncertainty
        neighbors_uncert = self.convinput_uncert(uncert)
        neigh_flat_uncert = neighbors.view(neighbors_uncert.size(0), -1)

        xcat = torch.cat((neigh_flat, neigh_flat_uncert), 1)
        neigh_flat_x = self.uncert(xcat)
        neigh_2d_x = neigh_flat_x.view(-1, 1, self.width, self.width)

        neighborsz = self.convinputz(neigh_z)
        neigh_flatz = neighborsz.view(neighborsz.size(0), -1)

        ## compute with uncertainty
        neighbors_uncertz = self.convinputz_uncert(uncert_z)
        neigh_flat_uncertz = neighbors.view(neighbors_uncertz.size(0), -1)

        zcat = torch.cat((neigh_flatz, neigh_flat_uncertz), 1)
        neigh_flat_z = self.uncertz(zcat)
        neigh_2d_z = neigh_flat_z.view(-1, 1, self.width, self.width)

        neighborsy = self.convinputy(neigh_y)
        neigh_flaty = neighborsy.view(neighborsy.size(0), -1)

        ## compute with uncertainty
        neighbors_uncerty = self.convinputy_uncert(uncert_y)
        neigh_flat_uncerty = neighbors.view(neighbors_uncerty.size(0), -1)

        ycat = torch.cat((neigh_flaty, neigh_flat_uncerty), 1)
        neigh_flat_y = self.uncerty(ycat) #169
        neigh_2d_y = neigh_flat_y.view(-1, 1, self.width, self.width)

        # feat_cat = torch.cat((neigh_flat_x, neigh_flat_z, neigh_flat_y), 1) #240

        # f = self.fgen(feat_cat)

        neighmod = self.convinput_mod(neigh_2d_x)
        neighzmod = self.convinputz_mod(neigh_2d_z)
        neighymod = self.convinputy_mod(neigh_2d_y)

        return neighmod, neighzmod, neighymod