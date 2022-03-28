import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

class model(nn.Module):

    def __init__(self, f1, f2, f3, kw1, kw2, kw3, pad=6, f_dim = 4, channels=3,
                 channels2 = 3, g=30):
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
        self.g = g

        #TODO: change the convolutions to AE? :)

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
        )
        self.convinput_max = nn.MaxPool2d(4, return_indices=True)
        self.convinput_max2 = nn.Sequential(
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
        )

        self.convinput_uncert_max = nn.MaxPool2d(4, return_indices=True)
        self.convinput_uncert_max2= nn.Sequential(
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
            nn.Linear(self.filters2, self.g),
            nn.BatchNorm1d(self.g),
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
        )
        self.convinputz_max = nn.MaxPool2d(4, return_indices=True)
        self.convinputz_max2 = nn.Sequential(
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
        )
        self.convinputz_uncert_max =nn.MaxPool2d(4, return_indices=True)
        self.convinputz_uncert_max2 = nn.Sequential(
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
            nn.Linear(self.filters2, self.g),
            nn.BatchNorm1d(self.g),
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
        )
        self.convinputy_max =nn.MaxPool2d(4, return_indices=True)
        self.convinputy_max2 = nn.Sequential(
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
        )
        self.convinputy_uncert_max =nn.MaxPool2d(4, return_indices=True)
        self.convinputy_uncert_max2 = nn.Sequential(
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
            nn.Linear(self.filters2, self.g),
            nn.BatchNorm1d(self.g),
            nn.LeakyReLU(0.1, True)
        )

        # f generation
        self.h_f1 = 20  # *2
        self.h_f2 = 30  # *2
        self.fgen = nn.Sequential(
            nn.Linear(self.g * 3, self.h_f1),
            nn.BatchNorm1d(self.h_f1),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f1, self.h_f2),
            nn.BatchNorm1d(self.h_f2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f2, self.f_dim)
        )

        ################################################################
        ################ decoder ################

        self.ifgen = nn.Sequential(
            nn.Linear(self.f_dim, self.h_f2),
            nn.BatchNorm1d(self.h_f2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f2, self.h_f1),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(self.h_f1),
            nn.Linear(self.h_f1,self.g * 3),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm1d(self.g * 3)
        )

        self.iuncerty = nn.Sequential(
            nn.Linear(self.g, self.filters2),
            nn.BatchNorm1d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.filters2, self.h_uncert2),
            nn.BatchNorm1d(self.h_uncert2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert2, self.h_uncert),
            nn.BatchNorm1d(self.h_uncert),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert, self.filters2 * 2)
        )
        self.iconvinputy_uncert_max = nn.MaxUnpool2d(4)
        self.iconvinputy_uncert_max2 = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
        )
        self.iconvinputy_uncert = nn.Sequential(
            nn.ConvTranspose2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters2, self.filters1, self.kw2),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters1, self.channels2, self.kw1, 1)
        )
        self.iconvinputy_max =nn.MaxUnpool2d(4)
        self.iconvinputy_max2 = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
        )
        self.iconvinputy = nn.Sequential(
            nn.ConvTranspose2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters2, self.filters1, self.kw2),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters1, self.channels, self.kw1, 1)
        )

        self.iuncertz = nn.Sequential(
            nn.Linear(self.g, self.filters2),
            nn.BatchNorm1d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.filters2, self.h_uncert2),
            nn.BatchNorm1d(self.h_uncert2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert2, self.h_uncert),
            nn.BatchNorm1d(self.h_uncert),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert, self.filters2 * 2)
        )

        self.iconvinputz_uncert_max =nn.MaxUnpool2d(4)
        self.iconvinputz_uncert_max2 = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
        )
        self.iconvinputz_uncert = nn.Sequential(
            nn.ConvTranspose2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters2, self.filters1, self.kw2),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters1, self.channels2, self.kw1, 1)
        )

        self.iconvinputz_max =nn.MaxUnpool2d(4)
        self.iconvinputz_max2 = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
        )
        self.iconvinputz = nn.Sequential(
            nn.ConvTranspose2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters2, self.filters1, self.kw2),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters1, self.channels, self.kw1, 1)
        )

        self.iuncert = nn.Sequential(
            nn.Linear(self.g, self.filters2),
            nn.BatchNorm1d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.filters2, self.h_uncert2),
            nn.BatchNorm1d(self.h_uncert2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert2, self.h_uncert),
            nn.BatchNorm1d(self.h_uncert),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_uncert, self.filters2 * 2)
        )

        self.iconvinput_uncert_max =nn.MaxUnpool2d(4)
        self.iconvinput_uncert_max2 = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
        )
        self.iconvinput_uncert = nn.Sequential(
            nn.ConvTranspose2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters2, self.filters1, self.kw2),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters1, self.channels2, self.kw1, 1)
        )

        self.iconvinput_max =nn.MaxUnpool2d(4)
        self.iconvinput_max2 = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
        )
        self.iconvinput = nn.Sequential(
            nn.ConvTranspose2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters2, self.filters1, self.kw2),
            nn.BatchNorm2d(self.filters1),
            nn.LeakyReLU(0.1, True),
            nn.ConvTranspose2d(self.filters1, self.channels, self.kw1, 1)
        )


    def forward(self, neigh, neigh_z, neigh_y, uncert, uncert_z, uncert_y):

        ################################################################
        ################ cycle 1 ################

        ################################################################
        ################ encoder ################

        neighbors = self.convinput(neigh)
        neighbors, indices_x = self.convinput_max(neighbors)
        neighbors = self.convinput_max2(neighbors)
        neigh_flat = neighbors.view(neighbors.size(0), -1)

        ## compute with uncertainty
        neighbors_uncert = self.convinput_uncert(uncert)
        neighbors_uncert, uindices_x = self.convinput_uncert_max(neighbors_uncert)
        neighbors_uncert = self.convinput_uncert_max2(neighbors_uncert)
        neigh_flat_uncert = neighbors.view(neighbors_uncert.size(0), -1)

        xcat = torch.cat((neigh_flat, neigh_flat_uncert), 1)
        neigh_flat_x = self.uncert(xcat)

        neighborsz = self.convinputz(neigh_z)
        neighborsz, indices_z = self.convinputz_max(neighborsz)
        neighborsz = self.convinputz_max2(neighborsz)
        neigh_flatz = neighborsz.view(neighborsz.size(0), -1)

        ## compute with uncertainty
        neighbors_uncertz = self.convinputz_uncert(uncert_z)
        neighbors_uncertz, uindices_z = self.convinputz_uncert_max(neighbors_uncertz)
        neighbors_uncertz = self.convinputz_uncert_max2(neighbors_uncertz)
        neigh_flat_uncertz = neighbors.view(neighbors_uncertz.size(0), -1)

        zcat = torch.cat((neigh_flatz, neigh_flat_uncertz), 1)
        neigh_flat_z = self.uncertz(zcat)

        neighborsy = self.convinputy(neigh_y)
        neighborsy, indices_y = self.convinputy_max(neighborsy)
        neighborsy = self.convinputy_max2(neighborsy)
        neigh_flaty = neighborsy.view(neighborsy.size(0), -1)

        ## compute with uncertainty
        neighbors_uncerty = self.convinputy_uncert(uncert_y)
        neighbors_uncerty, uindices_y = self.convinputy_uncert_max(neighbors_uncerty)
        neighbors_uncerty = self.convinputy_uncert_max2(neighbors_uncerty)
        neigh_flat_uncerty = neighbors.view(neighbors_uncerty.size(0), -1)

        ycat = torch.cat((neigh_flaty, neigh_flat_uncerty), 1)
        neigh_flat_y = self.uncerty(ycat) #169

        feat_cat = torch.cat((neigh_flat_x, neigh_flat_z, neigh_flat_y), 1)

        f = self.fgen(feat_cat)

        ################################################################
        ################ decoder ################

        #TODO: finish decoder
        ifeat_cat = self.ifgen(f)
        outputs = torch.split(ifeat_cat, self.g, dim=1)
        (ineigh_flat_x, ineigh_flat_z, ineigh_flat_y) = (outputs[0], outputs[1], outputs[2])

        iycat = self.iuncerty(ineigh_flat_y)
        outputs = torch.split(iycat,int(iycat.size(1)/2), dim=1)
        (ineigh_flaty, ineigh_flat_uncerty) = (outputs[0], outputs[1])
        ineigh_flat_uncerty = self.iconvinputy_uncert_max(ineigh_flat_uncerty.unsqueeze(2).unsqueeze(3), uindices_y)
        ineigh_flat_uncerty = self.iconvinputy_uncert_max2(ineigh_flat_uncerty)
        iuncerty = self.iconvinputy_uncert(ineigh_flat_uncerty)

        ineighborsy = self.iconvinputy_max(ineigh_flaty.unsqueeze(2).unsqueeze(3), indices_y)
        ineighborsy = self.iconvinputy_max2(ineighborsy)
        ineigh_y = self.iconvinputy(ineighborsy)


        izcat = self.iuncertz(ineigh_flat_z)
        outputs = torch.split(izcat, int(izcat.size(1)/2), dim=1)
        (ineigh_flatz, ineigh_flat_uncertz) = (outputs[0], outputs[1])
        ineigh_flat_uncertz = self.iconvinputz_uncert_max(ineigh_flat_uncertz.unsqueeze(2).unsqueeze(3), uindices_z)
        ineigh_flat_uncertz = self.iconvinputz_uncert_max2(ineigh_flat_uncertz)
        iuncertz = self.iconvinputz_uncert(ineigh_flat_uncertz)

        ineighborsz = self.iconvinputz_max(ineigh_flatz.unsqueeze(2).unsqueeze(3), indices_z)
        ineighborsz = self.iconvinputz_max2(ineighborsz)
        ineigh_z = self.iconvinputz(ineighborsz)

        ixcat = self.iuncertz(ineigh_flat_x)
        outputs = torch.split(ixcat, int(ixcat.size(1)/2), dim=1)
        (ineigh_flat, ineigh_flat_uncert) = (outputs[0], outputs[1])
        ineigh_flat_uncert = self.iconvinput_uncert_max(ineigh_flat_uncert.unsqueeze(2).unsqueeze(3), uindices_x)
        ineigh_flat_uncert = self.iconvinput_uncert_max2(ineigh_flat_uncert)
        iuncert = self.iconvinput_uncert(ineigh_flat_uncert)

        ineighbors = self.iconvinput_max(ineigh_flat.unsqueeze(2).unsqueeze(3), indices_x)
        ineighbors = self.iconvinput_max2(ineighbors)
        ineigh = self.iconvinput(ineighbors)

        # neighmod = self.convinput_mod(neigh_2d_x)
        # neighzmod = self.convinputz_mod(neigh_2d_z)
        # neighymod = self.convinputy_mod(neigh_2d_y)

        return ineigh, ineigh_z, ineigh_y, iuncert, iuncertz, iuncerty