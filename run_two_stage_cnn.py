import numpy as np
import torch, torchvision, os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
# import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import MRDataSet2_noupsample
import gc

processes = []

import two_stage_cnn
import importlib

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

init_seed = 1
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
np.random.seed(init_seed)

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


class Solver(object):
    def __init__(self, obj, valobj=None, epoch=30, batch_size=100000, lr=2e-4, d_dim_x=1, k_dim_x=3,
                 d_dim_n=1, k_dim_n=3,d_dim_t=1, k_dim_t=3, f_dim =5, pad =3,
                 beta=0.25, in_features=1, labels=3, shuffle=True, linepad=5, zlinepad=3, miscidx=None,
                 miscidx_val=None, params = None, channels=1):
        self.obj = obj
        self.valobj = valobj
        self.miscidx=miscidx
        self.miscidx_val = miscidx_val
        self.params = params
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.d_dim_x = d_dim_x
        self.k_dim_x = k_dim_x
        self.d_dim_n = d_dim_n
        self.k_dim_n = k_dim_n
        self.d_dim_t = d_dim_t
        self.k_dim_t = k_dim_t
        self.f_dim = f_dim
        self.pad = pad
        self.beta = beta
        self.in_features = in_features
        self.labels = labels
        self.shuffle = shuffle
        self.linepad = linepad
        self.zlinepad = zlinepad
        self.channels = channels

        ## TODO: add model_2input_mirrored
        self.model = MRDataSet2_noupsample.model_2input_mirrored(
            k_dim_x=self.k_dim_x,
            d_dim_x=self.d_dim_x,
            k_dim_n=self.k_dim_n,
            d_dim_n=self.d_dim_n,
            k_dim_t=self.k_dim_t,
            d_dim_t=self.d_dim_t,
            f_dim=self.f_dim,
            pad = self.pad,
            in_features=self.in_features,
            labels=self.labels,
            linepad=self.linepad,
            zlinepad=self.zlinepad,
            params=self.params,
            channels=self.channels)

        self.model = self.model.cuda()
        self.model.share_memory()

        # Criterions

        self.L1_Loss = nn.L1Loss().cuda()
        self.MSE_Loss = nn.MSELoss().cuda()

        self.CE_Loss = nn.CrossEntropyLoss().cuda()

        if self.miscidx is not None:
            multiinput = True
        else:
            multiinput = False

        # Dataset init
        self.data = MRDataSet2_noupsample.MRDataSet(pkl_file=self.obj,
                                           transform=transforms.Compose([
                                               MRDataSet2_noupsample.ToTensor(multiinput=multiinput)
                                           ]), miscidxs=self.miscidx,
                                                    multiinput=multiinput)


        self.dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=self.shuffle,
                                       num_workers=5, drop_last=False)

        if self.valobj:
            self.valdata = MRDataSet2_noupsample.MRDataSet(pkl_file=self.valobj,
                                                        transform=transforms.Compose([
                                                            MRDataSet2_noupsample.ToTensor(multiinput=multiinput)
                                                        ]), miscidxs=self.miscidx_val,
                                                    multiinput=multiinput)

            self.valdataloader = DataLoader(self.valdata, batch_size=self.batch_size, shuffle=self.shuffle,
                                         num_workers=5, drop_last=False)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))


    def set_mode(self, mode='train'):
        if mode == 'train':
            self.model.train()
        elif mode == 'eval':
            self.model.eval()
        else:
            raise ('mode error. It should be either train or eval')

    def train(self, **kwargs):
        self.set_mode('train')
        if 'epoch' in kwargs:
            self.epoch = kwargs['epoch']
        if 'lr' in kwargs:
            self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['lr'], betas=(0.5, 0.999))

        misclassified = []
        misclassified_val = []
        misclassified_softmax = []
        for e in range(self.epoch):
            print("Epoch {0}/{1}".format(e + 1, self.epoch))
            label_losses = []
            total_losses = []


            for idx, (sample, indices) in enumerate(self.dataloader):

                if self.miscidx is not None:
                    images1 = sample['image1_t1']
                    neighbors = sample['neighbors_t1']
                    neighbors_z = sample['neighbors_z_t1']
                    neighbors_y = sample['neighbors_y_t1']
                    ylabel = sample['label']

                else:

                    images1 = sample['image1']
                    neighbors = sample['neighbors']
                    neighbors_z = sample['neighbors_z']
                    neighbors_y = sample['neighbors_y']
                    ylabel = sample['label']

                X1 = Variable(images1.cuda(), requires_grad=False)
                neigh = Variable(neighbors.cuda(), requires_grad=False)
                neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                y = Variable(ylabel.cuda(), requires_grad=False)

                label_OHE, xhat = self.model(X1, neigh, neigh_z, neigh_y)


                label_loss = self.CE_Loss(label_OHE, y)


                total_loss = label_loss

                self.optimizer.zero_grad()
                total_loss.backward()

                self.optimizer.step()

                label_losses.append(label_loss.detach().data)
                total_losses.append(total_loss.detach().data)

                # TODO: test
                if idx != 0 and idx % 10 == 0:

                    # AVG Losses
                    label_losses_cat = torch.stack(label_losses, 0).mean()
                    total_losses_cat = torch.stack(total_losses, 0).mean()
                    print('\n[{:02d}/{:d}] label_loss:{:.2f} total_loss:{:.7f}'.format(
                        e+1,self.epoch, label_losses_cat, total_losses_cat))

                if e == self.epoch-1:
                    yhatidxs = torch.argmax(label_OHE, dim=1)
                    diff = yhatidxs - y
                    loc = diff.nonzero()
                    loc = loc.squeeze(1).data
                    misclassified.append(indices[loc].tolist())
                    misclassified_softmax.append(label_OHE.detach())

                del X1, neigh,images1, ylabel, sample, y

            # compute validation loss
            if self.valobj:
                label_losses = []
                total_losses = []
                with torch.no_grad():
                    for idx, (sample, indices) in enumerate(self.valdataloader):
                        if self.miscidx is not None:
                            images1 = sample['image1_t1']
                            neighbors = sample['neighbors_t1']
                            neighbors_z = sample['neighbors_z_t1']
                            neighbors_y = sample['neighbors_y_t1']
                            ylabel = sample['label']

                        else:

                            images1 = sample['image1']
                            # line = sample['line']
                            # zline = sample['zline']
                            neighbors = sample['neighbors']
                            neighbors_z = sample['neighbors_z']
                            neighbors_y = sample['neighbors_y']
                            ylabel = sample['label']

                        X1 = Variable(images1.cuda(), requires_grad=False)
                        neigh = Variable(neighbors.cuda(), requires_grad=False)
                        neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                        neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                        # Vert = Variable(line.cuda(), requires_grad=False)
                        # Zline = Variable(zline.cuda(), requires_grad=False)
                        y = Variable(ylabel.cuda(), requires_grad=False)

                        label_OHE, xhat = self.model(X1, neigh, neigh_z, neigh_y)


                        label_loss = self.CE_Loss(label_OHE, y)

                        if e == self.epoch - 1:
                            yhatidxs = torch.argmax(label_OHE, dim=1)
                            diff = yhatidxs - y
                            loc = diff.nonzero()
                            loc = loc.squeeze(1).data
                            misclassified_val.append(indices[loc].tolist())

                        total_loss = label_loss

                        label_losses.append(label_loss.detach().data)
                        total_losses.append(total_loss.detach().data)

                        # TODO: test
                        if idx != 0 and idx % 10 == 0:
                            # AVG Losses
                            label_losses_cat = torch.stack(label_losses, 0).mean()
                            total_losses_cat = torch.stack(total_losses, 0).mean()
                            print('\nVALIDATION [{:02d}/{:d}] label_loss:{:.2f} total_loss:{:.7f}'.format(
                                e + 1, self.epoch, label_losses_cat, total_losses_cat))

                        del X1, neigh, images1, ylabel, sample, y

        self.misclassified = misclassified
        self.misclassified_val = misclassified_val
        self.misclassified_softmax = misclassified_softmax

        print("[*] Training Finished!")


    def test(self, dataloader, orig_shape, test_data_hr_idx, hr_shape,d, second_model=None,
             miscidx=None, correction_model=None):
        self.set_mode('eval')
        self.model.eval()
        label_OHEs = []
        # self.Zs = []
        self.xhats = []
        self.Z_fs = []

        for idx, (sample, indices) in enumerate(dataloader):
            images1 = sample['image1']
            neighbors = sample['neighbors']
            neighbors_z = sample['neighbors_z']
            neighbors_y = sample['neighbors_y']

            X1 = Variable(images1.cuda(), requires_grad=False)
            neigh = Variable(neighbors.cuda(), requires_grad=False)
            neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
            neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)

            label_OHE, xhat = self.model(X1, neigh, neigh_z, neigh_y)

            if miscidx is not None:
                # idxs = np.array(list(set(indices) & set(miscidx)))
                test = indices.tolist()
                idxs = np.where(np.in1d(np.asarray(test) , miscidx))[0]
                # idxs = idxs

                if idxs.any():

                    images1 = sample['image1_t1']
                    neighbors = sample['neighbors_t1']
                    neighbors_z = sample['neighbors_z_t1']
                    neighbors_y = sample['neighbors_y_t1']

                    X1 = Variable(images1.cuda(), requires_grad=False)
                    neigh = Variable(neighbors.cuda(), requires_grad=False)
                    neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                    neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)

                    label_OHE_2, xhat = second_model.model(X1[idxs], neigh[idxs], neigh_z[idxs], neigh_y[idxs])
                    label_OHE[idxs] = label_OHE_2

            elif second_model is not None:
                tmp = np.sort(label_OHE.detach().data.cpu().numpy(), axis=1)
                diff = np.abs(tmp[:, -1] - tmp[:, -2])
                diffidx = np.where(diff < 0.3)

                if diffidx.any():

                    images1 = sample['image1_t1']
                    neighbors = sample['neighbors_t1']
                    neighbors_z = sample['neighbors_z_t1']
                    neighbors_y = sample['neighbors_y_t1']

                    X1 = Variable(images1.cuda(), requires_grad=False)
                    neigh = Variable(neighbors.cuda(), requires_grad=False)
                    neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                    neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)

                    label_OHE_2, xhat = second_model.model(X1[diffidx], neigh[diffidx], neigh_z[diffidx], neigh_y[diffidx])
                    label_OHE[diffidx] = label_OHE_2

            label_OHEs.append(label_OHE.detach())
            # self.Zs.append(Z_dec)
            self.xhats.append(xhat.detach())
            # self.Z_fs.append(Z_f.detach())


        labeled = torch.cat(label_OHEs, 0)
        labeled = labeled.data.cpu().numpy()



        labeled = np.argmax(labeled, axis=1)

        X = np.zeros(orig_shape)
        for i in np.arange(labeled.shape[0]):

            idxs = np.unravel_index(test_data_hr_idx[i], orig_shape)
            X[idxs] = labeled[i] + 1

        self.set_mode('train')

        return X


