import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np

class model(nn.Module):

    def __init__(self, f1, f2, f3, kw1, kw2, kw3, channels=1):
        super(model, self).__init__()

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        self.kw1 = kw1
        self.kw2 = kw2
        self.kw3 = kw3

        self.channels = channels

        self.convinput = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, 1, 0),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2, 1, 1),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f3, self.kw3, 1, 2),
            nn.BatchNorm2d(self.f3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f3, self.channels, self.kw1, 1, 0)
        )

        self.convinputz = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, 1, 0),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2, 1, 1),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f3, self.kw3, 1, 2),
            nn.BatchNorm2d(self.f3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f3, self.channels, self.kw1, 1, 0)
        )

        self.convinputy = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, 1, 0),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2, 1, 1),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f3, self.kw3, 1, 2),
            nn.BatchNorm2d(self.f3),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f3, self.channels, self.kw1, 1, 0)
        )

        #TODO: change the convolutions to discrete VAE? :)

    def forward(self, neigh, neigh_z, neigh_y):

        ################################################################
        ################ cycle 1 ################

        ################################################################
        ################ feature extraction ################

        neighmod = self.convinput(neigh)
        neighzmod = self.convinputz(neigh_z)
        neighymod = self.convinputy(neigh_y)


        ################################################################
        ################ classifier ################

        # [weightedImg, weightedImg_z, weightedImg_y] = self.createWImg(label_OHE)
        # modifiedImg = neigh + weightedImg
        # modifiedImg_z = neigh_z + weightedImg_z
        # modifiedImg_y = neigh_y + weightedImg_y
        #
        # ################################################################
        # ################ cycle 2 ################
        #
        # int_neighbors = self.convinput(modifiedImg)
        # neighbors = self.int_convinput(int_neighbors)
        # neigh_flat = neighbors.view(neighbors.size(0), -1)
        #
        # int_neighborsz = self.convinputz(modifiedImg_z)
        # neighborsz = self.int_convinputz(int_neighborsz)
        # neigh_flatz = neighborsz.view(neighborsz.size(0), -1)
        #
        # int_neighborsy = self.convinputy(modifiedImg_y)
        # neighborsy = self.int_convinputy(int_neighborsy)
        # neigh_flaty = neighborsy.view(neighborsy.size(0), -1)
        #
        # feat_cat = torch.cat((neigh_flat, neigh_flatz, neigh_flaty), 1)
        #
        # f = self.fgen(feat_cat)

        # _ , indices = label_OHE.max(1)

        return neighmod, neighzmod, neighymod

    # def hook(self, grad):
    #     self.grad_for_encoder = grad
    #     return grad

class model_1ch(nn.Module):

    def __init__(self, f1, f2, f3, kw1, kw2, kw3, channels=1):
        super(model_1ch, self).__init__()

        self.f1 = f1
        self.f2 = f2
        self.f3 = f3

        self.kw1 = kw1
        self.kw2 = kw2
        self.kw3 = kw3

        self.channels = channels

        self.convinput = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, 1, 0),
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

        self.convinputz = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, 1, 0),
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

        self.convinputy = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, 1, 0),
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

    def forward(self, neigh, neigh_z, neigh_y):

        ################################################################
        ################ cycle 1 ################

        ################################################################
        ################ feature extraction ################

        neighmod = self.convinput(neigh)
        neighzmod = self.convinputz(neigh_z)
        neighymod = self.convinputy(neigh_y)


        ################################################################
        ################ classifier ################

        # [weightedImg, weightedImg_z, weightedImg_y] = self.createWImg(label_OHE)
        # modifiedImg = neigh + weightedImg
        # modifiedImg_z = neigh_z + weightedImg_z
        # modifiedImg_y = neigh_y + weightedImg_y
        #
        # ################################################################
        # ################ cycle 2 ################
        #
        # int_neighbors = self.convinput(modifiedImg)
        # neighbors = self.int_convinput(int_neighbors)
        # neigh_flat = neighbors.view(neighbors.size(0), -1)
        #
        # int_neighborsz = self.convinputz(modifiedImg_z)
        # neighborsz = self.int_convinputz(int_neighborsz)
        # neigh_flatz = neighborsz.view(neighborsz.size(0), -1)
        #
        # int_neighborsy = self.convinputy(modifiedImg_y)
        # neighborsy = self.int_convinputy(int_neighborsy)
        # neigh_flaty = neighborsy.view(neighborsy.size(0), -1)
        #
        # feat_cat = torch.cat((neigh_flat, neigh_flatz, neigh_flaty), 1)
        #
        # f = self.fgen(feat_cat)

        # _ , indices = label_OHE.max(1)

        return neighmod, neighzmod, neighymod

    # def hook(self, grad):
    #     self.grad_for_encoder = grad
    #     return grad

