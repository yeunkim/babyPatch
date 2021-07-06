import numpy as np
# import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
import MRDataSet2_noupsample
import torch
from torch.utils.data import ConcatDataset
# from sklearn.neighbors import BallTree, KDTree
from torch.nn import DataParallel
import two_stage_cnn_uncertainty
import MRDataSet2_mult_dataset
import nibabel as nib
import itertools
import classify_weightedImg

processes = []

import two_stage_cnn
import importlib
from datetime import datetime
from truncatedloss import TruncatedLoss

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


class Solver(object):
    def __init__(self, obj, valobj=None, epoch=30, batch_size=100000, lr=2e-4, f_dim =5, pad =3,
                 beta=0.25, in_features=1, labels=3, shuffle=True, miscidx=None, valuncertfn = None,
                 miscidx_val=None, params = None, channels=1, coords=False, DL=False, testobjs=None,
                 width=3, softdiceloss = False, dropout=False, uncertainty = False, uncertfn = None,
                 channels2 = 3, iterative=False, textfn='/data/avgloss.txt', initmodel=None, initWImgmodel=None,
                 start_prune = 2, suffix = '', spherecoord=False):
        self.obj = obj
        self.valobj = valobj
        self.miscidx=miscidx
        self.miscidx_val = miscidx_val
        self.params = params
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.f_dim = f_dim
        self.pad = pad
        self.beta = beta
        self.in_features = in_features
        self.labels = labels
        self.shuffle = shuffle
        self.channels = channels
        self.coords = coords
        self.DLloss = DL
        self.testobjs = testobjs
        self.width = width
        self.diceloss = softdiceloss
        self.dropout = dropout
        self.uncertainty = uncertainty
        self.uncertfn = uncertfn
        self.channels2 = channels2
        self.iterative= iterative
        self.textfn = textfn
        self.initmodel = initmodel
        self.initWImgmodel = initWImgmodel
        self.start_prune = start_prune
        self.suffix = suffix
        self.valuncertfn = valuncertfn
        self.spherecoord = spherecoord

        if self.uncertainty:
            self.model = two_stage_cnn_uncertainty.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                in_features=self.in_features,
                labels=self.labels,
                params=self.params,
                channels=self.channels,
                channels2=self.channels2,
                spherecoords=self.spherecoord)

        else:
            self.model = two_stage_cnn.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                in_features=self.in_features,
                labels=self.labels,
                params=self.params,
                channels=self.channels,
                spherecoords=self.spherecoord)

        self.model = DataParallel(self.model)
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

        if not self.uncertainty:
            self.data =[]
            for i in np.arange(len(self.obj)):
                tmpdata = MRDataSet2_noupsample.MRDataSet(pkl_file=self.obj[i],
                                                   transform=transforms.Compose([
                                                       MRDataSet2_noupsample.ToTensor(multiinput=multiinput,
                                                                                      coords=self.coords,
                                                                                      spherecoord=self.spherecoord)
                                                   ]), miscidxs=self.miscidx, spherecoord=self.spherecoord,
                                                            multiinput=multiinput, coords=self.coords)
                self.data.append(tmpdata)

        elif self.uncertainty:
            self.data = []
            for i in np.arange(len(self.obj)):
                tmpdata = MRDataSet2_mult_dataset.MRDataSet(pkl_file=self.obj[i], pkl_file2=self.uncertfn[i],
                                                   transform=transforms.Compose([
                                                       MRDataSet2_mult_dataset.ToTensor(multiinput=multiinput,
                                                                                        coords=self.coords,
                                                                                        spherecoord=self.spherecoord)
                                                   ]), miscidxs=self.miscidx,spherecoord=self.spherecoord,
                                                            multiinput=multiinput, coords=self.coords)
                self.data.append(tmpdata)
        del tmpdata

        self.dataloader = DataLoader(ConcatDataset(self.data), batch_size=self.batch_size, shuffle=self.shuffle,
                                     num_workers=2, drop_last=False)

        if self.valobj:
            if not self.uncertainty:
                self.valdata = []
                for i in np.arange(len(self.valobj)):
                    tmpdata = MRDataSet2_noupsample.MRDataSet(pkl_file=self.valobj[i],
                                                                transform=transforms.Compose([
                                                                    MRDataSet2_noupsample.ToTensor(
                                                                        spherecoord=self.spherecoord)
                                                                ]), )
                    self.valdata.append(tmpdata)

                self.valdataloader = DataLoader(ConcatDataset(self.valdata), batch_size=self.batch_size,
                                                shuffle=self.shuffle,
                                                num_workers=5, drop_last=False)
            elif self.uncertainty:
                self.valdata = []
                for i in np.arange(len(self.valobj)):
                    tmpdata = MRDataSet2_mult_dataset.MRDataSet(pkl_file=self.valobj[i], pkl_file2=self.valuncertfn[i],
                                                                transform=transforms.Compose([
                                                                    MRDataSet2_mult_dataset.ToTensor(spherecoord=self.spherecoord)
                                                                ]), spherecoord=self.spherecoord)
                    self.valdata.append(tmpdata)

                self.valdataloader = DataLoader(ConcatDataset(self.valdata), batch_size=self.batch_size,
                                                shuffle=self.shuffle,
                                                num_workers=5, drop_last=False)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # truncated loss criterion
        self.criterion = TruncatedLoss(trainset_size=len(ConcatDataset(self.data))).cuda()
        self.valcriterion = TruncatedLoss(trainset_size=len(ConcatDataset(self.valdata))).cuda()

    def reset_model(self):
        self.model = two_stage_cnn.model_2input_mirrored(
            f_dim=self.f_dim,
            pad=self.pad,
            in_features=self.in_features,
            labels=self.labels,
            params=self.params,
            channels=self.channels,
            spherecoords=self.spherecoord)

        self.model = DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.share_memory()
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # truncated loss criterion
        self.criterion = TruncatedLoss(trainset_size=len(ConcatDataset(self.data))).cuda()
        self.valcriterion = TruncatedLoss(trainset_size=len(ConcatDataset(self.valdata))).cuda()

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

        terminate = 0
        width = (2 * self.width) + 1
        prev_acc = 0
        val_prev_acc = 0

        restart = True
        while restart:
            restart = False
            for e in range(self.epoch):
                print("Epoch {0}/{1}".format(e + 1, self.epoch))
                label_losses = []
                train_loss = 0
                correct = 0
                total = 0
                val_train_loss = 0
                val_correct = 0
                val_total = 0

                for idx, (sample, indices, orig_indices, indx, indz, indy) in enumerate(self.dataloader):
                    if self.uncertainty:
                        neighbors = sample['neighbors']
                        neighbors_z = sample['neighbors_z']
                        neighbors_y = sample['neighbors_y']
                        ylabel = sample['label']
                        neighbors2 = sample['neighbors2']
                        neighbors2_z = sample['neighbors2_z']
                        neighbors2_y = sample['neighbors2_y']

                        if self.spherecoord:
                            spherecoord = sample['spherecoord']
                            spherecoordinates = Variable(spherecoord.cuda(), requires_grad=False)
                        else:
                            spherecoordinates = None

                        neigh = Variable(neighbors.cuda(), requires_grad=False)
                        neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                        neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                        y = Variable(ylabel.cuda(), requires_grad=False)
                        neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                        neigh2_z = Variable(neighbors2_z.cuda(), requires_grad=False)
                        neigh2_y = Variable(neighbors2_y.cuda(), requires_grad=False)

                        label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y, neigh2, neigh2_z, neigh2_y,
                                                              spherecoord=spherecoordinates)

                    else:
                        neighbors = sample['neighbors']
                        neighbors_z = sample['neighbors_z']
                        neighbors_y = sample['neighbors_y']
                        ylabel = sample['label']

                        if self.spherecoord:
                            spherecoord = sample['spherecoord']
                            spherecoordinates = Variable(spherecoord.cuda(), requires_grad=False)
                        else:
                            spherecoordinates = None

                        neigh = Variable(neighbors.cuda(), requires_grad=False)
                        neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                        neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                        y = Variable(ylabel.cuda(), requires_grad=False)

                        label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y, spherecoord=spherecoordinates)

                    truncloss =  self.criterion(label_OHE, y, indices) # self.CE_Loss(label_OHE, y)
                    self.optimizer.zero_grad()
                    truncloss.backward()
                    self.optimizer.step()

                    label_losses.append(truncloss.detach().data)

                    train_loss += truncloss.item()
                    _, predicted = torch.max(label_OHE.data, 1)
                    total += y.size(0)
                    correct += predicted.eq(y.data).cpu().sum()
                    correct = correct.item()

                    if idx != 0 and idx % 10 == 0:
                        txt = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (idx + 1), 100. * correct / total, correct, total)
                        print(txt)
                        label_losses_mod_cat = 0
                        label_losses_cat = torch.stack(label_losses, 0).mean()
                        with open(self.textfn, 'a+') as f:
                            f.write("{0}\t{1}\n".format(label_losses_cat, label_losses_mod_cat))

                if e ==2 and not self.uncertainty:
                    if label_losses_cat > 0:
                        restart = True
                        self.reset_model()
                        break

                if self.valobj:
                    self.set_mode('eval')
                    print("Validation:")
                    with torch.no_grad():
                        for idx, (sample, indices, orig_indices, indx, indz, indy) in enumerate(self.valdataloader):
                            if self.uncertainty:
                                neighbors = sample['neighbors']
                                neighbors_z = sample['neighbors_z']
                                neighbors_y = sample['neighbors_y']
                                ylabel = sample['label']
                                neighbors2 = sample['neighbors2']
                                neighbors2_z = sample['neighbors2_z']
                                neighbors2_y = sample['neighbors2_y']

                                if self.spherecoord:
                                    spherecoord = sample['spherecoord']
                                    spherecoordinates = Variable(spherecoord.cuda(), requires_grad=False)
                                else:
                                    spherecoordinates = None

                                neigh = Variable(neighbors.cuda(), requires_grad=False)
                                neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                                neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                                y = Variable(ylabel.cuda(), requires_grad=False)
                                neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                                neigh2_z = Variable(neighbors2_z.cuda(), requires_grad=False)
                                neigh2_y = Variable(neighbors2_y.cuda(), requires_grad=False)

                                label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y, neigh2, neigh2_z, neigh2_y,
                                                                      spherecoord=spherecoordinates)

                            else:
                                neighbors = sample['neighbors']
                                neighbors_z = sample['neighbors_z']
                                neighbors_y = sample['neighbors_y']
                                ylabel = sample['label']

                                if self.spherecoord:
                                    spherecoord = sample['spherecoord']
                                    spherecoordinates = Variable(spherecoord.cuda(), requires_grad=False)
                                else:
                                    spherecoordinates = None

                                neigh = Variable(neighbors.cuda(), requires_grad=False)
                                neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                                neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                                y = Variable(ylabel.cuda(), requires_grad=False)

                                label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y, spherecoord=spherecoordinates)

                            truncloss = self.valcriterion(label_OHE, y, indices)

                            val_train_loss += truncloss.item()
                            _, predicted = torch.max(label_OHE.data, 1)
                            val_total += y.size(0)
                            val_correct += predicted.eq(y.data).cpu().sum()
                            val_correct = val_correct.item()

                            if idx != 0 and idx % 10 == 0:
                                txt = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                                val_train_loss / (idx + 1), 100. * val_correct / val_total, val_correct, val_total)
                                print(txt)

                        print("Validation end")
                delta = 0.01
                if e > 2 and not self.uncertainty:
                    if prev_acc > (correct/total)+delta:
                        terminate +=1
                    elif terminate >0 :
                        terminate -= 1
                    if val_prev_acc > (val_correct / val_total) + delta:
                        terminate += 1
                    elif terminate >0 :
                        terminate -= 1
                    prev_acc = correct / total
                    val_prev_acc = val_correct / val_total
                    if terminate > 2:
                        break
                else:
                    val_prev_acc = val_correct / val_total
                    prev_acc = correct / total

        print("[*] Training Finished!")

    def test(self, intimgname, intoutname, affine, dataloader= None, batchsize =1000, imgs = None, second_model=None,
             miscidx=None, correction_model=None, modifiedimgs=None, uncertfn = None):
        self.set_mode('eval')
        self.model.eval()
        X_list = []
        labeled_list = []
        self.xhat_list = []
        if dataloader is not None:
            datanum = 1
        elif imgs is not None:
            datanum = len(imgs)
        else:
            datanum = len(self.obj)

        for d in np.arange(datanum):
            self.xhats = []
            self.maxindxs = []
            label_OHEs = []
            self.maxindxs_mod = []
            if (dataloader is None) and (imgs is None):
                ind_dataloader = DataLoader(self.data[d], batch_size=batchsize, shuffle=False,
                                             num_workers=6, drop_last=False)
                size = self.data[d].dataset.dataOrigShape[:3]
                origindices = self.data[d].dataset.indices
            elif imgs is not None:
                if not self.uncertainty:
                    data = []
                    for i in np.arange(datanum):
                        tmpdata = MRDataSet2_noupsample.MRDataSet(pkl_file=imgs[i],
                                                                  transform=transforms.Compose([
                                                                      MRDataSet2_noupsample.ToTensor(
                                                                          coords=self.coords,
                                                                          spherecoord=self.spherecoord)
                                                                  ]), miscidxs=self.miscidx,
                                                                  spherecoord=self.spherecoord,
                                                                  coords=self.coords)
                        data.append(tmpdata)

                elif self.uncertainty:
                    data = []
                    for i in np.arange(datanum):
                        tmpdata = MRDataSet2_mult_dataset.MRDataSet(pkl_file=imgs[i], pkl_file2=uncertfn[i],
                                                                    transform=transforms.Compose([
                                                                        MRDataSet2_mult_dataset.ToTensor(
                                                                            coords=self.coords,
                                                                            spherecoord=self.spherecoord)
                                                                    ]), miscidxs=self.miscidx,
                                                                    spherecoord=self.spherecoord,coords=self.coords)
                        data.append(tmpdata)
                del tmpdata
                ind_dataloader = DataLoader(data[d], batch_size=batchsize, shuffle=False,
                                            num_workers=6, drop_last=False)
                size = data[d].dataset.dataOrigShape[:3]
                origindices = data[d].dataset.indices

            elif dataloader is not None:
                ind_dataloader = dataloader
                size = dataloader.dataset.dataset.dataOrigShape[:3]
                origindices = dataloader.dataset.dataset.indices

            for idx, (sample, indices, orig_indices, indx, indz, indy) in enumerate(ind_dataloader):

                if self.uncertainty:
                    neighbors = sample['neighbors']
                    neighbors_z = sample['neighbors_z']
                    neighbors_y = sample['neighbors_y']
                    ylabel = sample['label']
                    neighbors2 = sample['neighbors2']
                    neighbors2_z = sample['neighbors2_z']
                    neighbors2_y = sample['neighbors2_y']

                    neigh = Variable(neighbors.cuda(), requires_grad=False)
                    neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                    neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                    y = Variable(ylabel.cuda(), requires_grad=False)
                    neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                    neigh2_z = Variable(neighbors2_z.cuda(), requires_grad=False)
                    neigh2_y = Variable(neighbors2_y.cuda(), requires_grad=False)

                    label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y, neigh2, neigh2_z, neigh2_y)

                else:
                    neighbors = sample['neighbors']
                    neighbors_z = sample['neighbors_z']
                    neighbors_y = sample['neighbors_y']

                    neigh = Variable(neighbors.cuda(), requires_grad=False)
                    neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                    neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                    label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y)

                label_OHEs.append(label_OHE.detach())

                self.xhats.append(xhat.detach())
                self.maxindxs.append(maxindx.detach().data)

            labeled = torch.cat(label_OHEs, 0)
            labeled = labeled.data.cpu().numpy()
            maxindices = torch.cat(self.maxindxs, 0).cpu().numpy()

            X = np.zeros(size)
            b = np.asarray(list(itertools.chain.from_iterable(self.xhats)))
            values = np.zeros([b.shape[0], self.f_dim])
            for B in np.arange(len(b)):
                values[B] = b[B].data.cpu().numpy()

            size4d = size + (self.f_dim,)
            Y = np.zeros(size4d)
            size3d = size + (self.labels,)
            L = np.zeros(size3d)

            for idx in np.arange(maxindices.shape[0]):
                idxs = np.unravel_index(origindices[idx], size)
                X[idxs] = maxindices[idx] + 1
                Y[idxs] = values[idx]
                L[idxs] = labeled[idx]

            recon = nib.Nifti1Image(Y, affine=affine[d])
            nib.save(recon, filename=intimgname[d])

            recon = nib.Nifti1Image(X, affine=affine[d])
            nib.save(recon, filename=intoutname[d])

            labeled_list.append(L.reshape(np.prod(size3d[:3]), 3))

        return labeled_list



