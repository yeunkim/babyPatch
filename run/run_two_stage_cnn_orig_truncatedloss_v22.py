import numpy as np
# import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from Dataset import MRDataSet2_noupsample, MRDataSet2_mult_dataset, MRDataSet_v22, MRDataSet_mult_v22
import torch
from torch.utils.data import ConcatDataset
# from sklearn.neighbors import BallTree, KDTree
from torch.nn import DataParallel
from models import two_stage_cnn_uncertainty, two_stage_cnn
import nibabel as nib
import itertools
import h5py

processes = []

from truncatedloss import TruncatedLoss

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


class Solver(object):
    def __init__(self, obj, valobj=None, epoch=30, batch_size=100000, lr=2e-4, f_dim =5, pad =5, lossThresh = 0.2,
                 beta=0.25, labels=3, shuffle=True, valuncertfn = None, numslices =(10,0,0),
                 channels=1, uncertainty = False, uncertfn = None, slices = False, axes = (True,True,True),
                 channels2 = 3, textfn='/data/avgloss.txt', suffix = '', dataset_portion = 0.5
                 ):
        self.obj = obj
        self.valobj = valobj
        self.epoch = epoch
        self.batch_size = batch_size
        self.lr = lr
        self.f_dim = f_dim
        self.pad = pad
        self.beta = beta
        self.labels = labels
        self.shuffle = shuffle
        self.channels = channels
        self.uncertainty = uncertainty
        self.uncertfn = uncertfn
        self.channels2 = channels2
        self.slices = slices
        self.axes = axes
        self.numslices = numslices
        self.textfn = textfn
        self.suffix = suffix
        self.valuncertfn = valuncertfn
        self.lossThresh = lossThresh
        self.dataset_portion = dataset_portion
        if self.uncertainty:
            self.model = two_stage_cnn_uncertainty.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                labels=self.labels,
                channels=self.channels,
                channels2=self.channels2)
        else:
            self.model = two_stage_cnn.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                labels=self.labels,
                channels=self.channels)

        self.model = DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.share_memory()

        # Criterions

        self.L1_Loss = nn.L1Loss().cuda()
        self.MSE_Loss = nn.MSELoss().cuda()

        self.CE_Loss = nn.CrossEntropyLoss().cuda()
        self.data = []
        if not self.uncertainty:
            for i in np.arange(len(self.obj)):
                tmpdata = MRDataSet_v22.MRDataSet(file=self.obj[i],
                                                  transform=transforms.Compose([MRDataSet_v22.ToTensor()]),
                                                  render=True, slices=self.slices, axes=self.axes,
                                                  numslices=self.numslices)
                self.data.append(tmpdata)
        elif self.uncertainty:
            for i in np.arange(len(self.obj)):
                tmpdata = MRDataSet_mult_v22.MRDataSet(file=self.obj[i], file2=self.uncertfn[i],
                                                    transform=transforms.Compose([MRDataSet_mult_v22.ToTensor()]),
                                                  render=True, slices=self.slices, axes=self.axes,
                                                  numslices=self.numslices)
                self.data.append(tmpdata)
        del tmpdata
        self.traintotalAmount = ConcatDataset(self.data).cumulative_sizes[-1]
        self.trainAmount = int(self.traintotalAmount * self.dataset_portion)

        traindata, testdata = torch.utils.data.random_split(ConcatDataset(self.data),
                                                            [int(self.trainAmount), int(self.traintotalAmount-self.trainAmount)])
        self.dataloader = DataLoader(traindata, batch_size=self.batch_size, shuffle=self.shuffle,
                                     num_workers=2, drop_last=False)

        if self.valobj:
            self.valdata = []
            if not self.uncertainty:
                for i in np.arange(len(self.valobj)):
                    tmpdata = MRDataSet_v22.MRDataSet(file=self.valobj[i],
                                                      transform=transforms.Compose([MRDataSet_v22.ToTensor()]),
                                                      render=True, slices=self.slices, axes=self.axes,
                                                      numslices=self.numslices)
                    self.valdata.append(tmpdata)
            elif self.uncertainty:
                for i in np.arange(len(self.valobj)):
                    tmpdata = MRDataSet_mult_v22.MRDataSet(file=self.valobj[i], file2=self.valuncertfn[i],
                                                        transform=transforms.Compose([MRDataSet_mult_v22.ToTensor()]),
                                                  render=True, slices=self.slices, axes=self.axes,
                                                  numslices=self.numslices)
                    self.valdata.append(tmpdata)
            del tmpdata
            self.valtotalAmount = ConcatDataset(self.valdata).cumulative_sizes[0]
            self.valtrainAmount = int(self.valtotalAmount * self.dataset_portion)
            valdata, _ = torch.utils.data.random_split(ConcatDataset(self.valdata),
                                                       [int(self.valtrainAmount), int(self.valtotalAmount - self.valtrainAmount)])
            self.valdataloader = DataLoader(valdata, batch_size=self.batch_size,
                                            shuffle=self.shuffle, num_workers=5, drop_last=False)
            self.valcriterion = TruncatedLoss(trainset_size=self.valtotalAmount).cuda()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # truncated loss criterion
        self.criterion = TruncatedLoss(trainset_size=self.traintotalAmount).cuda()

    def reset_model(self):
        if self.uncertainty:
            self.model = two_stage_cnn_uncertainty.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                labels=self.labels,
                channels=self.channels,
                channels2=self.channels2)
        else:
            self.model = two_stage_cnn.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                labels=self.labels,
                channels=self.channels)

        self.model = DataParallel(self.model)
        self.model = self.model.cuda()
        self.model.share_memory()
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # truncated loss criterion
        self.criterion = TruncatedLoss(trainset_size=self.traintotalAmount).cuda()
        self.valcriterion = TruncatedLoss(trainset_size=self.valtotalAmount).cuda()

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
        prev_acc = 0
        val_prev_acc = 0

        restart = True
        while restart:
            restart = False
            for e in range(self.epoch):
                print("Epoch {0}/{1}".format(e + 1, self.epoch))
                for idx, (sample, indices, orig_indices, _) in enumerate(self.dataloader):
                    label_losses = []
                    train_loss = 0
                    correct = 0
                    total = 0
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
                        ylabel = sample['label']

                        neigh = Variable(neighbors.cuda(), requires_grad=False)
                        neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                        neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                        y = Variable(ylabel.cuda(), requires_grad=False)

                        label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y)

                    truncloss =  self.criterion(label_OHE, y, indices)
                    self.optimizer.zero_grad()
                    truncloss.backward()
                    self.optimizer.step()

                    label_losses.append(truncloss.detach().data)

                    train_loss += truncloss.item()
                    _, predicted = torch.max(label_OHE.data, 1)
                    total += y.size(0)
                    correct += predicted.eq(y.data).cpu().sum()
                    correct = correct.item()

                    # if idx != 0 and idx % 10 == 0:
                    txt = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (idx + 1), 100. * correct / total, correct, total)
                    print(txt)
                    label_losses_mod_cat = 0
                    label_losses_cat = torch.stack(label_losses, 0).mean()
                    print('avg loss:', label_losses_cat)
                    with open(self.textfn, 'a+') as f:
                        f.write("{0}\t{1}\n".format(label_losses_cat, label_losses_mod_cat))

                ## TODO: figure out what's happening with the loss
                # if e ==2 and not self.uncertainty:
                #     label_losses_cat = torch.stack(label_losses, 0).mean()
                #     if label_losses_cat > self.lossThresh:
                #         restart = True
                #         self.reset_model()
                #         break

                if self.valobj:
                    self.set_mode('eval')
                    print("Validation:")
                    with torch.no_grad():
                        for idx, (sample, indices, orig_indices, _) in enumerate(self.valdataloader):
                            val_train_loss = 0
                            val_correct = 0
                            val_total = 0
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
                                ylabel = sample['label']

                                neigh = Variable(neighbors.cuda(), requires_grad=False)
                                neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                                neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                                y = Variable(ylabel.cuda(), requires_grad=False)

                                label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y)

                            truncloss = self.valcriterion(label_OHE, y, indices)

                            val_train_loss += truncloss.item()
                            _, predicted = torch.max(label_OHE.data, 1)
                            val_total += y.size(0)
                            val_correct += predicted.eq(y.data).cpu().sum()
                            val_correct = val_correct.item()

                            # if idx != 0 and idx % 10 == 0:
                            txt = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                            val_train_loss / (idx + 1), 100. * val_correct / val_total, val_correct, val_total)
                            print(txt)

                        print("Validation end")

                ## TODO: implement termination criteria

        print("[*] Training Finished!")

    def test(self, intimgname, intoutname, affine, dataloader= None, batchsize =1000, imgs = None, uncertfn = None,
             slices = False, axes = (True, True, True), numslices = (10,0,0)):
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
                    data = MRDataSet_v22.MRDataSet(file=imgs[d],
                                                      transform=transforms.Compose([MRDataSet_v22.ToTensor()]),
                                                      render=True,
                                                   slices=slices, axes=axes,
                                                      numslices=numslices
                                                   )
                else:
                    data = MRDataSet_mult_v22.MRDataSet(file=imgs[d], file2=uncertfn[d],
                                                        transform=transforms.Compose(
                                                            [MRDataSet_mult_v22.ToTensor()]),
                                                        render=True, slices=slices, axes=axes,
                                                        numslices=numslices)

                ind_dataloader = DataLoader(data, batch_size=batchsize, shuffle=False,
                                            num_workers=6, drop_last=False)
                with h5py.File(imgs[d], "r") as f:
                    size = f.attrs['origsize'][:3]
                    bounds = f.attrs['bounds']
                    padsize = f['data'].shape
                # del data
                origindices = []

            elif dataloader is not None:
                ind_dataloader = dataloader
                size = dataloader.dataset.dataset.dataOrigShape[:3]
                origindices = dataloader.dataset.dataset.indices

            for idx, (sample, indices, orig_indices, _) in enumerate(ind_dataloader):

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

                origindices.append(orig_indices)

                label_OHEs.append(label_OHE.detach())

                self.xhats.append(xhat.detach())
                self.maxindxs.append(maxindx.detach().data)

            del ind_dataloader
            origindices = np.concatenate(origindices)
            labeled = torch.cat(label_OHEs, 0)
            del label_OHEs
            labeled = labeled.data.cpu().numpy()
            maxindices = torch.cat(self.maxindxs, 0).cpu().numpy()
            maxindices = maxindices + 1
            del self.maxindxs


            self.xhats = np.asarray(list(itertools.chain.from_iterable(self.xhats)))
            values = np.zeros([self.xhats.shape[0], self.f_dim])
            for B in np.arange(len(self.xhats)):
                values[B] = self.xhats[B].data.cpu().numpy()
            del self.xhats

            size4d = np.concatenate((padsize, [self.f_dim]))
            size3d = np.concatenate((padsize, [self.labels]))

            Y = np.zeros(size4d)
            Y.reshape((np.prod(padsize), self.f_dim))[origindices] = values
            # remove padding
            Y = Y[self.pad:-self.pad, self.pad:-self.pad,self.pad:-self.pad]
            recon = nib.Nifti1Image(Y, affine=affine[d])
            del Y, values
            nib.save(recon, filename=intimgname[d])
            del recon

            X = np.zeros(padsize)
            X.ravel()[origindices] = maxindices
            X = X[self.pad:-self.pad, self.pad:-self.pad, self.pad:-self.pad]
            recon = nib.Nifti1Image(X.astype(np.int16), affine=affine[d])
            # recon.header.set_data_dtype(np.int16)
            del X

            nib.save(recon, filename=intoutname[d])
            del recon

            L = np.zeros(size3d)
            L.reshape((np.prod(padsize), self.labels))[origindices] = labeled
            L = L[self.pad:-self.pad, self.pad:-self.pad, self.pad:-self.pad]
            labeled_list.append(L.reshape(np.prod(size3d[:3]), 3))

            del L
            del orig_indices

        return labeled_list



