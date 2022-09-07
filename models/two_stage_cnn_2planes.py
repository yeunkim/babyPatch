import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

class model_2input_mirrored(nn.Module):

    def __init__(self,
                labels=4, pad=3, f_dim = 5, channels=3,
                 spherecoords=None):
        super(model_2input_mirrored, self).__init__()
        self.channels = channels
        self.f_dim = f_dim
        self.pad =pad
        self.labels = labels
        self.h1 = 5 *2
        self.h2 = 10 *2
        self.h2b = 5 *3
        self.spherecoords = spherecoords

        def conv2d_size_out(size, kernel_size, stride=1, pad=0):
            return int(((size - kernel_size + (2 * pad)) / stride) + 1)

        def compute_pad(input_size, kernel_size):
            padsize = int(kernel_size / 2)
            return int(padsize - (input_size % kernel_size))

        width = (2*pad)+1
        self.width = width
        self.height = width
        self.kw1 = int(width / 5)
        self.pad1 = compute_pad(width, self.kw1)
        osize1 = conv2d_size_out(width, self.kw1, pad=self.pad1)
        self.kw2 = int(osize1 / 4)
        self.pad2 = compute_pad(osize1, self.kw2)
        osize2 = conv2d_size_out(osize1, self.kw2, pad=self.pad2)
        self.kw3 = int(osize2 / 3)
        self.pad3 = compute_pad(osize2, self.kw3)
        osize3 = conv2d_size_out(osize2, self.kw3, pad=self.pad3)
        self.kw4 = int(osize3 / 2)
        self.pad4 = compute_pad(osize3, self.kw4)
        osize4 = conv2d_size_out(osize3, self.kw4, pad=self.pad4)


        # N1 input
        self.filters1 = 48
        self.filters2 = 72
        self.filters3 = 84
        self.filters4 = 108

        self.convinput = nn.Sequential(
            nn.Conv2d(self.channels, self.filters1, self.kw1, padding=self.pad1),
            nn.BatchNorm2d(self.filters1),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2, padding=self.pad2),
            nn.BatchNorm2d(self.filters2),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters3, self.kw3, padding=self.pad3),
            nn.BatchNorm2d(self.filters3),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters3, self.filters4, self.kw4, padding=self.pad4),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(self.filters4)
            # nn.MaxPool2d(4)
        )

        self.int_convinput = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        # Nz1 input
        # self.convinputz = nn.Sequential(
        #     nn.Conv2d(self.channels, self.filters1, self.kw1, 1),
        #     nn.BatchNorm2d(self.filters1),
        #     # nn.Dropout2d(0.5),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(self.filters1, self.filters2, self.kw2),
        #     nn.BatchNorm2d(self.filters2),
        #     # nn.Dropout2d(0.5),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv2d(self.filters2, self.filters2, self.kw3),
        #     nn.BatchNorm2d(self.filters2),
        #     # nn.Dropout2d(0.5),
        #     nn.LeakyReLU(0.1, True),
        #     nn.MaxPool2d(4)
        # )

        self.convinputz = nn.Sequential(
            nn.Conv2d(self.channels, self.filters1, self.kw1, padding=self.pad1),
            nn.BatchNorm2d(self.filters1),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2, padding=self.pad2),
            nn.BatchNorm2d(self.filters2),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters3, self.kw3, padding=self.pad3),
            nn.BatchNorm2d(self.filters3),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters3, self.filters4, self.kw4, padding=self.pad4),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(self.filters4)
            # nn.MaxPool2d(4)
        )

        # self.int_convinputz = nn.Sequential(
        #     nn.BatchNorm2d(self.filters2),
        #     nn.LeakyReLU(0.1, True)
        # )

        # Ny1 input
        self.convinputy = nn.Sequential(
            nn.Conv2d(self.channels, self.filters1, self.kw1, 1),
            nn.BatchNorm2d(self.filters1),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters1, self.filters2, self.kw2),
            nn.BatchNorm2d(self.filters2),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.filters2, self.filters2, self.kw3),
            nn.BatchNorm2d(self.filters2),
            # nn.Dropout2d(0.5),
            nn.LeakyReLU(0.1, True),
            nn.MaxPool2d(4)
        )

        self.int_convinputy = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        # self.convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.width, self.kw1), self.kw2), self.kw3)
        # self.convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(self.height, self.kw1), self.kw2), self.kw3)
        self.linear_input_size = int(osize4 * osize4 * 2 * self.filters4)

        # f generation
        self.h_f1 = 40 # *2
        self.h_f2 = 50 # *2
        self.h_f3 = 60
        self.fgen = nn.Sequential(
            # nn.Linear(self.d_dim_x + self.d_dim_n + self.d_dim_t, self.h_f1),
            # nn.Linear( self.filters2*3 + self.prez, self.h_f1),
            nn.Linear(self.linear_input_size, self.h_f1),
            nn.BatchNorm1d( self.h_f1),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f1, self.h_f2),
            nn.BatchNorm1d(self.h_f2),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f2, self.h_f3),
            nn.BatchNorm1d(self.h_f3),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f3, self.f_dim)
        )

        self.h3 = 12 #*2
        self.h4 = 6 #*2


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
                # nn.Dropout(0.5),
                nn.LeakyReLU(0.1, True),
                # nn.Linear(self.h3, self.h3),
                # nn.BatchNorm1d(self.h3),
                # nn.LeakyReLU(0.1, True),
                nn.Linear(self.h3, self.h4),
                nn.BatchNorm1d(self.h4),
                # nn.Dropout(0.5),
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

    def forward(self, neigh1, neigh2, spherecoord=None):
        neighbors = self.convinput(neigh1.float())
        # feat_neigh = int_neighbors.view(int_neighbors.size(0), -1)
        # neighbors = self.int_convinput(int_neighbors)
        neigh_flat = neighbors.view(neighbors.size(0), -1)

        neighborsz = self.convinputz(neigh2.float())
        # feat_neighz = int_neighborsz.view(int_neighborsz.size(0), -1)
        # neighborsz = self.int_convinputz(int_neighborsz)
        neigh_flatz = neighborsz.view(neighborsz.size(0), -1)

        feat_cat = torch.cat((neigh_flat, neigh_flatz), 1)

        f = self.fgen(feat_cat)

        ################################################################
        ################ decoder ################
        ## add spherical coordinate (append here)
        if self.spherecoords:
            f_spherecoord = torch.cat((f,spherecoord.view(-1,3).float()), 1)
            label_OHE = self.labelpredic_spherecoord(f_spherecoord)
        else:
            label_OHE = self.labelpredic(f)

        _ , indices = label_OHE.max(1)

        return label_OHE, f , indices

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad
