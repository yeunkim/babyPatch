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
        # self.convflatdim = int(7*((pad*self.d))**2)
        # self.convflatdim = int(self.filters2 * (5 ** 2))
        # self.convflatdim = int(self.filters2*1)
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


        # x input
        # self.input = nn.Sequential(
        #     nn.Linear(self.in_features, self.h1),
        #     nn.BatchNorm1d(self.h1),
        #     nn.LeakyReLU(0.1, True),
        #     # nn.Linear(self.h1, self.h1),
        #     # nn.BatchNorm1d(self.h1),
        #     # nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.h1, self.prez)
        #
        # )

        # self.encode = nn.Sequential(
        #     nn.BatchNorm1d(self.prez),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(int(self.prez*1), self.h2),
        #     nn.BatchNorm1d(self.h2),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.h2, self.h2b),
        #     nn.BatchNorm1d(self.h2b),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.h2b,self.d_dim_x)
        # )

        # N1 input
        self.filters1 = 10 * 2 *2 # *2
        self.filters2 = 20 * 2 *2 # *2
        self.kw1 = 2
        self.kw2 = 3
        self.kw3 = 5
        self.convflatdim = self.filters2

        self.convinput = nn.Sequential(
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

        self.int_convinput = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        # self.reduceconv = nn.Sequential(
        #
        #     # nn.MaxPool2d(int((pad*self.d)/2)),
        #     nn.Linear(self.convflatdim, 24),
        #     nn.BatchNorm1d(24),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(24, 12),
        #     nn.BatchNorm1d(12),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(12, self.d_dim_n)
        # )

        # Nz1 input
        self.convinputz = nn.Sequential(
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

        self.int_convinputz = nn.Sequential(
            nn.BatchNorm2d(self.filters2),
            nn.LeakyReLU(0.1, True)
        )

        # self.reduceconvz = nn.Sequential(
        #
        #     # nn.MaxPool2d(int((pad*self.d)/2)),
        #     nn.Linear(self.convflatdim, 24),
        #     nn.BatchNorm1d(24),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(24, 12),
        #     nn.BatchNorm1d(12),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(12, self.d_dim_n)
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

        # self.reduceconvy = nn.Sequential(
        #
        #     # nn.MaxPool2d(int((pad*self.d)/2)),
        #     nn.Linear(self.convflatdim, 24),
        #     nn.BatchNorm1d(24),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(24, 12),
        #     nn.BatchNorm1d(12),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(12, self.d_dim_n)
        # )

        # N2 input
        # self.lineinput = nn.Sequential(
        #     nn.Conv1d(1, self.numf1, self.convsize1),
        #     nn.BatchNorm1d(self.numf1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Conv1d(self.numf1, self.numf2, self.convsize2)
        #
        # )
        #
        # self.linemaxpool =nn.MaxPool1d(self.lineout, return_indices=True)
        #
        # self.int_lineinput = nn.Sequential(
        #     nn.BatchNorm1d(self.numf2),
        #     nn.LeakyReLU(0.1, True)
        # )
        #
        # self.linereduce = nn.Sequential(
        #
        #     nn.Linear(self.numf2, self.lineh),
        #     nn.BatchNorm1d(self.lineh),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.lineh, self.d_dim_t)
        # )


        # f generation
        self.h_f1 = 20 # *2
        self.h_f2 = 30 # *2
        self.fgen = nn.Sequential(
            # nn.Linear(self.d_dim_x + self.d_dim_n + self.d_dim_t, self.h_f1),
            # nn.Linear( self.filters2*3 + self.prez, self.h_f1),
            nn.Linear(self.filters2 * 3, self.h_f1),
            nn.BatchNorm1d( self.h_f1),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f1, self.h_f2),
            nn.BatchNorm1d(self.h_f2),
            # nn.Dropout(0.5),
            nn.LeakyReLU(0.1, True),
            nn.Linear(self.h_f2, self.f_dim)
        )

        # self.h_noise = 8
        # self.noise = nn.Sequential(
        #     nn.Linear(self.f_dim+self.channels, self.h_noise),
        #     nn.BatchNorm1d(self.h_noise),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.h_noise, self.f_dim)
        #
        # )

        self.h3 = 12 # *2
        self.h4 = 6 # *2

        # classification
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


        # Embedding book for x
        # self.embd_x = nn.Embedding(self.k_dim_x,self.d_dim_x ).cuda()
        #
        # # Embedding book for n
        # self.embd_n = nn.Embedding(self.k_dim_n,self.d_dim_n ).cuda()
        #
        # # Embedding book for nz
        # self.embd_nz = nn.Embedding(self.k_dim_n, self.d_dim_n).cuda()
        #
        # # Embedding book for ny
        # self.embd_ny = nn.Embedding(self.k_dim_n, self.d_dim_n).cuda()
        #
        # # Embedding book for t
        # self.embd_t = nn.Embedding(self.k_dim_t, self.d_dim_t ).cuda()

        # Embedding book for f
        # self.embd_f = nn.Embedding(self.k_dim_n, self.f_dim).cuda()


        # f decode
        # self.fdec = nn.Sequential(
        #     nn.Linear(self.f_dim, self.h_f2),
        #     nn.BatchNorm1d(self.h_f2),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.h_f2, self.h_f1),
        #     nn.BatchNorm1d(self.h_f1),
        #     nn.LeakyReLU(0.1, True),
        #     # nn.Linear(self.h_f1, self.d_dim_x + self.d_dim_n + self.d_dim_t)
        #     nn.Linear(self.h_f1, self.d_dim_n + self.d_dim_n + self.d_dim_x + self.d_dim_n)
        # )
        #
        # # decode x
        # self.decode_input = nn.Sequential(
        #     nn.Linear(self.d_dim_x,self.h2b),
        #     nn.BatchNorm1d(self.h2b),
        #     nn.LeakyReLU(0.1,True),
        #     nn.Linear(self.h2b, self.h2),
        #     nn.BatchNorm1d(self.h2),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.h2, int(self.prez*1)),
        #     nn.BatchNorm1d(int(self.prez*1)),
        #     nn.LeakyReLU(0.1,True)
        #
        # )
        #
        # self.decode1 = nn.Sequential(
        #     # nn.Linear(15, 5),
        #     # nn.BatchNorm1d(5),
        #     # nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.prez, self.h1),
        #     nn.BatchNorm1d(self.h1),
        #     nn.LeakyReLU(0.1,True),
        #     # nn.Linear(self.h1, self.h1),
        #     # nn.BatchNorm1d(self.h1),
        #     # nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.h1, self.in_features),
        #     # nn.BatchNorm1d(self.in_features),
        #     # nn.Tanh()
        # )
        #
        # # decode n
        # self.decode_reduceconv = nn.Sequential(
        #     nn.Linear(self.d_dim_n, 12),
        #     nn.BatchNorm1d(12),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(12, 24),
        #     nn.BatchNorm1d(24),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(24, self.convflatdim)
        #
        # )
        #
        # self.int_decode_reduceconv = nn.Sequential(
        #     nn.BatchNorm1d(self.convflatdim),
        #     nn.LeakyReLU(0.1, True)
        # )
        #
        # self.decode_convinput = nn.Sequential(
        #
        #     nn.ConvTranspose2d(self.filters2,self.filters1, int(self.neighsize / self.d)),
        #     nn.BatchNorm2d(self.filters1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.ConvTranspose2d(self.filters1, 1, self.d, stride=self.d),
        #     # nn.BatchNorm2d(1),
        #     # nn.LeakyReLU(0.1, True)
        # )
        #
        # # decode nz
        # self.decode_reduceconvz = nn.Sequential(
        #     nn.Linear(self.d_dim_n, 12),
        #     nn.BatchNorm1d(12),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(12, 24),
        #     nn.BatchNorm1d(24),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(24, self.convflatdim)
        #
        # )
        #
        # self.int_decode_reduceconvz = nn.Sequential(
        #     nn.BatchNorm1d(self.convflatdim),
        #     nn.LeakyReLU(0.1, True)
        # )
        #
        # self.decode_convinputz = nn.Sequential(
        #
        #     nn.ConvTranspose2d(self.filters2, self.filters1, int(self.neighsize / self.d)),
        #     nn.BatchNorm2d(self.filters1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.ConvTranspose2d(self.filters1, 1, self.d, stride=self.d),
        #     # nn.BatchNorm2d(1),
        #     # nn.LeakyReLU(0.1, True)
        # )
        #
        # # decode ny
        # self.decode_reduceconvy = nn.Sequential(
        #     nn.Linear(self.d_dim_n, 12),
        #     nn.BatchNorm1d(12),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(12, 24),
        #     nn.BatchNorm1d(24),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(24, self.convflatdim)
        #
        # )
        #
        # self.int_decode_reduceconvy = nn.Sequential(
        #     nn.BatchNorm1d(self.convflatdim),
        #     nn.LeakyReLU(0.1, True)
        # )
        #
        # self.decode_convinputy = nn.Sequential(
        #
        #     nn.ConvTranspose2d(self.filters2, self.filters1, int(self.neighsize / self.d)),
        #     nn.BatchNorm2d(self.filters1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.ConvTranspose2d(self.filters1, 1, self.d, stride=self.d),
        #     # nn.BatchNorm2d(1),
        #     # nn.LeakyReLU(0.1, True)
        # )
        #
        # # decode t
        # self.decode_linereduce = nn.Sequential(
        #     nn.Linear(self.d_dim_t, self.lineh),
        #     nn.BatchNorm1d(self.lineh),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(self.lineh, self.numf2)
        #
        # )
        #
        # self.int_decode_linereduce = nn.Sequential(
        #     nn.BatchNorm1d(self.numf2),
        #     nn.LeakyReLU(0.1, True)
        # )
        #
        # self.linemaxunpool = nn.MaxUnpool1d(self.lineout)
        #
        #
        # self.decode_lineinput = nn.Sequential(
        #     nn.BatchNorm1d(self.numf2),
        #     nn.LeakyReLU(0.1, True),
        #     nn.ConvTranspose1d(self.numf2, self.numf1, self.convsize2),
        #     nn.BatchNorm1d(self.numf1),
        #     nn.LeakyReLU(0.1, True),
        #     nn.ConvTranspose1d(self.numf1, 1, self.convsize1)
        # )


    def find_nearest(self,query,target):
        Q=query.unsqueeze(1).repeat(1,target.size(0),1)
        T=target.unsqueeze(0).repeat(query.size(0),1,1)
        index=(Q-T).pow(2).sum(2).sqrt().min(1)[1]
        return target[index]

    def forward(self, X1, neigh, neigh_z, neigh_y):

        ################################################################
        ################ cycle forward ################

        ################################################################
        ################ encoder ################

        # if self.params is not None:
        #     X_enc1 = self.params.input(X1.view(-1, self.in_features))
        #
        #     int_neighbors = self.params.convinput(neigh)
        #     feat_neigh = int_neighbors.view(int_neighbors.size(0), -1)
        #     neighbors = self.params.int_convinput(int_neighbors)
        #     neigh_flat = neighbors.view(neighbors.size(0), -1)
        #
        #     int_neighborsz = self.params.convinputz(neigh_z)
        #     feat_neighz = int_neighborsz.view(int_neighborsz.size(0), -1)
        #     neighborsz = self.params.int_convinputz(int_neighborsz)
        #     neigh_flatz = neighborsz.view(neighborsz.size(0), -1)
        #
        #     int_neighborsy = self.params.convinputy(neigh_y)
        #     feat_neighy = int_neighborsy.view(int_neighborsy.size(0), -1)
        #     neighborsy = self.params.int_convinputy(int_neighborsy)
        #     neigh_flaty = neighborsy.view(neighborsy.size(0), -1)
        #
        #     feat_cat = torch.cat((X_enc1, neigh_flat, neigh_flatz, neigh_flaty), 1)
        #
        #     f = self.params.fgen(feat_cat)
        #
        # else:
        # X_enc1 = self.input(X1.view(-1,self.in_features))

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

        # int_vertconv0= self.lineinput(line)
        # int_vertconv , indices = self.linemaxpool(int_vertconv0)
        # feat_vert = int_vertconv.view(int_vertconv.size(0), -1)
        # vertconv = self.int_lineinput(int_vertconv)
        # vertconv_flat = vertconv.view(vertconv.size(0), -1)

        # Z_enc_x = self.encode(X_enc1)
        #
        # Z_enc_n = self.reduceconv(neigh_flat)
        #
        # Z_enc_nz = self.reduceconvz(neigh_flatz)
        #
        # Z_enc_ny = self.reduceconvy(neigh_flaty)
        #
        # # Z_enc_t = self.linereduce(vertconv_flat)
        #
        # Z_dec_x = self.find_nearest(Z_enc_x, self.embd_x.weight)
        # Z_dec_x.register_hook(self.hook)
        #
        # Z_dec_n = self.find_nearest(Z_enc_n, self.embd_n.weight)
        # Z_dec_n.register_hook(self.hook)
        #
        # Z_dec_nz = self.find_nearest(Z_enc_nz, self.embd_nz.weight)
        # Z_dec_nz.register_hook(self.hook)
        #
        # Z_dec_ny = self.find_nearest(Z_enc_ny, self.embd_ny.weight)
        # Z_dec_ny.register_hook(self.hook)

        # Z_dec_t = self.find_nearest(Z_enc_t, self.embd_t.weight)
        # Z_dec_t.register_hook(self.hook)

        # Z_dec_cat = torch.cat((Z_dec_x, Z_dec_n, Z_dec_t), 1)
        # Z_dec_cat = torch.cat((Z_dec_n,Z_dec_nz, Z_dec_t, Z_dec_x, Z_dec_ny), 1)
        # Z_dec_cat = torch.cat((Z_dec_n, Z_dec_nz, Z_dec_x, Z_dec_ny), 1)
        # feat_cat = torch.cat((X_enc1, neigh_flat, neigh_flatz, neigh_flaty),1)
        feat_cat = torch.cat((neigh_flat, neigh_flatz, neigh_flaty), 1)

        f = self.fgen(feat_cat)
        # f.register_hook(self.hook)
        # Z_f = self.find_nearest(f, self.embd_f.weight)
        # Z_f.register_hook(self.hook)

        # f_for_embd = self.find_nearest(self.embd_f.weight, f)

        # f_cat = torch.cat((f, X1.view(-1,self.channels)), 1)
        #
        # f = self.noise(f)

        ################################################################
        ################ decoder ################

        label_OHE = self.labelpredic(f)
        # varlabel_OHE = Variable(label_OHE.data, requires_grad=True)
        # label_OHE.register_hook(self.hook)

        _ , indices = label_OHE.max(1)
        # value.register_hook(self.hook)

        ################################################################
        ################ cycle backward ################

        # Z_dec_cat = self.fdec(f)
        # # Z_split = torch.split(Z_dec_cat, [self.d_dim_x, self.d_dim_n, self.d_dim_t], dim=1)
        # # Z_split = torch.split(Z_dec_cat, [self.d_dim_n, self.d_dim_n, self.d_dim_t, self.d_dim_x, self.d_dim_n], dim=1)
        # Z_split = torch.split(Z_dec_cat, [self.d_dim_n, self.d_dim_n, self.d_dim_x, self.d_dim_n], dim=1)
        #
        # # z_back_x = self.decode_input(Z_split[0])
        # # feat_back_n = self.decode_reduceconv(Z_split[1])
        # # feat_back_t = self.decode_linereduce(Z_split[2])
        #
        # feat_back_n = self.decode_reduceconv(Z_split[0])
        # feat_back_nz = self.decode_reduceconvz(Z_split[1])
        # # feat_back_t = self.decode_linereduce(Z_split[2])
        # z_back_x = self.decode_input(Z_split[2])
        # feat_back_ny = self.decode_reduceconvy(Z_split[3])
        #
        # X_recon1 = self.decode1(z_back_x).view(-1,1,self.orig_img_size,self.orig_img_size)
        #
        # int_neigh_back = self.int_decode_reduceconv(feat_back_n)
        # neigh_back = int_neigh_back.view(-1, self.filters2, 1, 1)
        # N_dec1 = self.decode_convinput(neigh_back).view(-1, 1, neigh.size(2), neigh.size(2))
        #
        # # int_line_back = self.int_decode_linereduce(feat_back_t)
        # # line_back0 = int_line_back.view(-1, self.numf2, 1)
        # # line_back = self.linemaxunpool(line_back0, indices)
        # # T_dec1 = self.decode_lineinput(line_back)
        # #
        # # T_dec1 = T_dec1.view(-1, line.size(1))
        #
        # # X_recon1, X_enc1, z_back_x, Z_enc_x, Z_dec_x = (0,0,0,0,0)
        # feat_vert, feat_back_t, Z_enc_t, Z_dec_t, Z_enc_t_for_embd, T_dec1 = (0,0,0,0,0,0)
        #
        # ################################################################
        # ################ commitment loss ################
        #
        # Z_enc_x_for_embd = self.find_nearest(self.embd_x.weight, Z_enc_x)
        # Z_enc_n_for_embd = self.find_nearest(self.embd_n.weight, Z_enc_n)
        # Z_enc_nz_for_embd = self.find_nearest(self.embd_nz.weight, Z_enc_nz)
        # Z_enc_ny_for_embd = self.find_nearest(self.embd_ny.weight, Z_enc_ny)
        # # Z_enc_t_for_embd = self.find_nearest(self.embd_t.weight, Z_enc_t)
        #
        # # Z_enc_x_for_embd =0
        # Z_enc_t_for_embd = 0

        # return X_recon1, X_enc1, z_back_x, feat_neigh, feat_back_n,feat_vert, feat_back_t, Z_enc_x, Z_enc_n, Z_enc_t, \
        #        Z_dec_x, Z_dec_n, Z_dec_t, Z_enc_x_for_embd, Z_enc_n_for_embd, Z_enc_t_for_embd, label_OHE , N_dec1, \
        #        T_dec1, feat_neighz, feat_back_nz, Z_enc_nz, Z_dec_nz, Z_enc_nz_for_embd, feat_neighy, feat_back_ny, \
        #        Z_enc_ny, Z_dec_ny, Z_enc_ny_for_embd
        return label_OHE, f , indices #, Z_f ,f_for_embd

    def hook(self, grad):
        self.grad_for_encoder = grad
        return grad
