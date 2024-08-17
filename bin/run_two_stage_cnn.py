import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
# import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import MRDataSet2_noupsample
from scipy.spatial import cKDTree
import torch
from torch.utils.data import ConcatDataset
from sklearn.neighbors import BallTree, KDTree
from torch.nn import DataParallel
import two_stage_cnn_dropout
import two_stage_cnn_uncertainty
import MRDataSet2_mult_dataset
import nibabel as nib
import itertools

processes = []

import two_stage_cnn
import importlib

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# init_seed = 1
# torch.manual_seed(init_seed)
# torch.cuda.manual_seed(init_seed)
# np.random.seed(init_seed)

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


class Solver(object):
    def __init__(self, obj, valobj=None, epoch=30, batch_size=100000, lr=2e-4, f_dim =5, pad =3,
                 beta=0.25, in_features=1, labels=3, shuffle=True, miscidx=None,
                 miscidx_val=None, params = None, channels=1, coords=False, DL=False, testobjs=None,
                 width=None, softdiceloss = False, dropout=False, uncertainty = False, uncertfn = None,
                 channels2 = 3):
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


        if self.dropout:
            self.model = two_stage_cnn_dropout.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                in_features=self.in_features,
                labels=self.labels,
                params=self.params,
                channels=self.channels)
        elif self.uncertainty:
            self.model = two_stage_cnn_uncertainty.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                in_features=self.in_features,
                labels=self.labels,
                params=self.params,
                channels=self.channels,
                channels2=self.channels2)
        else:
            self.model = two_stage_cnn.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                in_features=self.in_features,
                labels=self.labels,
                params=self.params,
                channels=self.channels)

        self.model = self.model.cuda()
        self.model.share_memory()

        self.model = DataParallel(self.model)

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
                                                       MRDataSet2_noupsample.ToTensor(multiinput=multiinput, coords=self.coords)
                                                   ]), miscidxs=self.miscidx,
                                                            multiinput=multiinput, coords=self.coords)
                self.data.append(tmpdata)

        elif self.uncertainty:
            self.data = []
            for i in np.arange(len(self.obj)):
                tmpdata = MRDataSet2_mult_dataset.MRDataSet(pkl_file=self.obj[i], pkl_file2=self.uncertfn[i],
                                                   transform=transforms.Compose([
                                                       MRDataSet2_mult_dataset.ToTensor(multiinput=multiinput, coords=self.coords)
                                                   ]), miscidxs=self.miscidx,
                                                            multiinput=multiinput, coords=self.coords)
                self.data.append(tmpdata)
        del tmpdata

        self.dataloader = DataLoader(ConcatDataset(self.data), batch_size=self.batch_size, shuffle=self.shuffle,
                                     num_workers=2, drop_last=False)

        if self.valobj:
            self.valdata = MRDataSet2_noupsample.MRDataSet(pkl_file=self.valobj,
                                                        transform=transforms.Compose([
                                                            MRDataSet2_noupsample.ToTensor(multiinput=multiinput,coords=self.coords)
                                                        ]), miscidxs=self.miscidx_val,
                                                    multiinput=multiinput,coords=self.coords)

            self.valdataloader = DataLoader(self.valdata, batch_size=self.batch_size, shuffle=self.shuffle,
                                         num_workers=5, drop_last=False)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))


    def dice_loss(self, output, target, weights=None, ignore_index=None):
        """
        output : NxCxHxW Variable
        target :  NxHxW LongTensor
        weights : C FloatTensor
        ignore_index : int index to ignore from loss
        """
        eps = 0.000001
        encoded_target = output.detach() * 0
        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        # else:
        encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        # if ignore_index is not None:
        #     denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)

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

        width = (2 * self.width) + 1
        # ktree = cKDTree(np.expand_dims(self.dataloader.dataset.dataset.indices, 1))
        if self.diceloss:
            balltree = KDTree(np.expand_dims(self.dataloader.dataset.dataset.indices, 1))

        for e in range(self.epoch):
            print("Epoch {0}/{1}".format(e + 1, self.epoch))
            label_losses = []
            total_losses = []


            for idx, (sample, indices, orig_indices) in enumerate(self.dataloader):

                if self.miscidx is not None:
                    neighbors = sample['neighbors_t1']
                    neighbors_z = sample['neighbors_z_t1']
                    neighbors_y = sample['neighbors_y_t1']
                    ylabel = sample['label']

                elif self.uncertainty:
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

                label_loss = self.CE_Loss(label_OHE, y)

                total_loss = label_loss
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                # total_loss.backward()

                if self.diceloss:
                    ## image patch bases
                    imgpatchbase = np.vstack([np.arange(-1*self.width, self.width + 1)] * (width))
                    imgpatchcoeff = np.array([np.arange(-1*self.width, self.width + 1),] * (width)).transpose()
                    coeff = np.prod(self.dataloader.dataset.dataset.dataOrigShape[1:])
                    coeff_y = self.dataloader.dataset.dataset.dataOrigShape[2]
                    coeff_x = np.prod(self.dataloader.dataset.dataset.dataOrigShape[1:])

                    tmp = np.repeat(orig_indices.detach(), (width)**2).view((-1, (width), ((2*self.width) +1)))
                    imgpatches = coeff * torch.from_numpy(imgpatchbase) + torch.from_numpy(imgpatchcoeff) + tmp

                    ## TODO: makes imgpatches for coeff_y
                    imgpatches_y = coeff_y * torch.from_numpy(imgpatchbase) + torch.from_numpy(imgpatchcoeff) + tmp

                    ## TODO: makes imgpatches for coeff_x
                    imgpatches_x = (coeff_x * torch.from_numpy(imgpatchbase)) + (coeff_y* torch.from_numpy(imgpatchcoeff)) + tmp

                    del imgpatchbase, imgpatchcoeff, tmp

                    patchindices = imgpatches.view(-1).numpy()
                    patchindices_y = imgpatches_y.view(-1).numpy()
                    # patchindices_x = imgpatches_x.view(-1).numpy()
                    # patchindices = np.concatenate([patchindices, patchindices_x, patchindices_y])
                    patchindices = np.concatenate([patchindices, patchindices_y])
                    del imgpatches, patchindices_y # , patchindices_x
                    patchindices = np.expand_dims(patchindices, 1)
                    # dd, ii = ktree.query_ball_tree(patchindices, 2)
                    _, ii = balltree.query(patchindices, k=1)
                    del patchindices

                    self.data.transform = None
                    subsetdata = torch.utils.data.Subset(self.data, ii)
                    dataloader2 = DataLoader(subsetdata, batch_size=100*((2*self.width) +1)**2, shuffle=False, num_workers=1)
                    # dataloader2.dataset.transform = None
                    diceloss = 0

                    for idx2, (sample2, indices2, orig_indices2) in enumerate(dataloader2):
                    # for i in range(0, len(subsetdata)):
                        if self.channels == 1:
                            neighbors2 = sample2['neighbors'] #.transpose((2, 0, 1)).astype('float32') transpose(4,1).view(-1, 4, 11, 11)
                            neighbors_z2 = sample2['neighbors_z'] #.transpose((2, 0, 1)).astype('float32')
                            neighbors_y2 = sample2['neighbors_y'] #.transpose((2, 0, 1)).astype('float32')
                            ylabel2 = sample2['label']
                        else:
                            neighbors2 = sample2['neighbors'].transpose(4,1).view(-1, self.channels, 11, 11)
                            neighbors_z2 = sample2['neighbors_z'].transpose(4,1).view(-1, self.channels, 11, 11)
                            neighbors_y2 = sample2['neighbors_y'].transpose(4,1).view(-1, self.channels, 11, 11)
                            ylabel2 = sample2['label']

                        neigh2 = Variable(neighbors2.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
                        neigh_z2 = Variable(neighbors_z2.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
                        neigh_y2 = Variable(neighbors_y2.cuda(), requires_grad=False).type(torch.cuda.FloatTensor)
                        y2 = Variable(ylabel2.cuda(), requires_grad=False).type(torch.cuda.LongTensor)

                        label_OHE2, xhat2, maxindx2 = self.model(neigh2, neigh_z2, neigh_y2)

                        diceloss = 0.5 * self.dice_loss(label_OHE2.view(-1, self.labels, width, width), y2.view(-1, width, width))
                        # diceloss = self.diceloss(maxindx2.view((-1, width, width)),
                        #                                     y2.view((-1, width, width)))

                        # diceloss.backward(retain_graph=True)
                        # diceloss2 = diceloss.repeat((width**2)) #.view(-1)
                        # total_loss = total_loss + 0.5*diceloss
                        diceloss.backward()
                        # total_loss.backward()
                        self.optimizer.step()
                        # tmp = self.model.grad_for_encoder
                        # diceloss = diceloss.unsqueeze(1).repeat(1, 49).view(-1).unsqueeze(1).repeat(1,3).type(torch.cuda.FloatTensor)
                        # label_OHE2.backward(diceloss)
                else:
                    self.optimizer.step()
                    diceloss = total_loss

                ## pass back through f with the features
                # xhat.backward(self.model.grad_for_encoder)

                if self.DLloss:
                    DN = torch.mean(torch.abs(torch.mm(xhat, torch.t(xhat)) - torch.eye(xhat.size(0)).cuda()))
                    loss2 = 0.3* DN
                    loss2.backward()

                # self.optimizer.step()

                if self.diceloss:
                    self.data.transform = transforms.Compose([MRDataSet2_noupsample.ToTensor()])

                label_losses.append(label_loss.detach().data)
                total_losses.append(diceloss.detach().data)

                # TODO: test
                if idx != 0 and idx % 10 == 0:

                    # AVG Losses
                    label_losses_cat = torch.stack(label_losses, 0).mean()
                    total_losses_cat = torch.stack(total_losses, 0).mean()
                    print('\n[{:02d}/{:d}] label_loss:{:.2f} dice_loss:{:.7f}'.format(
                        e+1,self.epoch, label_losses_cat, total_losses_cat))

                if e == self.epoch-1:
                    # self.label_OHE = label_OHE
                    yhatidxs = torch.argmax(label_OHE, dim=1)
                    diff = yhatidxs - y
                    loc = diff.nonzero()
                    loc = loc.squeeze(1).data
                    misclassified.append(indices[loc].tolist())
                    misclassified_softmax.append(label_OHE.detach())


                del neigh, ylabel, sample, y
            # compute validation loss
            # if self.valobj:
            #     label_losses = []
            #     total_losses = []
            #     with torch.no_grad():
            #         for idx, (sample, indices) in enumerate(self.valdataloader):
            #             if self.miscidx is not None:
            #                 images1 = sample['image1_t1']
            #                 neighbors = sample['neighbors_t1']
            #                 neighbors_z = sample['neighbors_z_t1']
            #                 neighbors_y = sample['neighbors_y_t1']
            #                 ylabel = sample['label']
            #
            #             else:
            #
            #                 images1 = sample['image1']
            #                 # line = sample['line']
            #                 # zline = sample['zline']
            #                 neighbors = sample['neighbors']
            #                 neighbors_z = sample['neighbors_z']
            #                 neighbors_y = sample['neighbors_y']
            #                 ylabel = sample['label']
            #
            #             X1 = Variable(images1.cuda(), requires_grad=False)
            #             neigh = Variable(neighbors.cuda(), requires_grad=False)
            #             neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
            #             neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
            #             # Vert = Variable(line.cuda(), requires_grad=False)
            #             # Zline = Variable(zline.cuda(), requires_grad=False)
            #             y = Variable(ylabel.cuda(), requires_grad=False)
            #
            #             label_OHE, xhat = self.model(X1, neigh, neigh_z, neigh_y)
            #
            #
            #             label_loss = self.CE_Loss(label_OHE, y)
            #
            #             if e == self.epoch - 1:
            #                 yhatidxs = torch.argmax(label_OHE, dim=1)
            #                 diff = yhatidxs - y
            #                 loc = diff.nonzero()
            #                 loc = loc.squeeze(1).data
            #                 misclassified_val.append(indices[loc].tolist())
            #
            #             total_loss = label_loss
            #
            #             label_losses.append(label_loss.detach().data)
            #             total_losses.append(total_loss.detach().data)
            #
            #             # TODO: test
            #             if idx != 0 and idx % 10 == 0:
            #                 # AVG Losses
            #                 label_losses_cat = torch.stack(label_losses, 0).mean()
            #                 total_losses_cat = torch.stack(total_losses, 0).mean()
            #                 print('\nVALIDATION [{:02d}/{:d}] label_loss:{:.2f} total_loss:{:.7f}'.format(
            #                     e + 1, self.epoch, label_losses_cat, total_losses_cat))
            #
            #             del X1, neigh, images1, ylabel, sample, y

        self.misclassified = misclassified
        self.misclassified_val = misclassified_val
        self.misclassified_softmax = misclassified_softmax

        print("[*] Training Finished!")

    def test(self, intimgname, intoutname, affine, dataloader= None, batchsize =1000, second_model=None,
             miscidx=None, correction_model=None):
        self.set_mode('eval')
        self.model.eval()
        X_list = []
        labeled_list = []
        self.xhat_list = []
        if dataloader is not None:
            datanum = 1
        else:
            datanum = len(self.obj)

        for i in np.arange(datanum):
            self.xhats = []
            self.maxindxs = []
            label_OHEs = []
            if dataloader is None:
                ind_dataloader = DataLoader(self.data[i], batch_size=batchsize, shuffle=False,
                                             num_workers=3, drop_last=False)
            else:
                ind_dataloader = dataloader

            for idx, (sample, indices, orig_indices) in enumerate(ind_dataloader):

                if miscidx is not None:
                    # idxs = np.array(list(set(indices) & set(miscidx)))
                    test = indices.tolist()
                    idxs = np.where(np.in1d(np.asarray(test) , miscidx))[0]
                    # idxs = idxs

                    if idxs.any():
                        neighbors = sample['neighbors_t1']
                        neighbors_z = sample['neighbors_z_t1']
                        neighbors_y = sample['neighbors_y_t1']

                        neigh = Variable(neighbors.cuda(), requires_grad=False)
                        neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                        neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)

                        label_OHE_2, xhat = second_model.model(neigh[idxs], neigh_z[idxs], neigh_y[idxs])
                        label_OHE[idxs] = label_OHE_2

                elif second_model is not None:
                    tmp = np.sort(label_OHE.detach().data.cpu().numpy(), axis=1)
                    diff = np.abs(tmp[:, -1] - tmp[:, -2])
                    diffidx = np.where(diff < 0.3)

                    if diffidx.any():
                        neighbors = sample['neighbors_t1']
                        neighbors_z = sample['neighbors_z_t1']
                        neighbors_y = sample['neighbors_y_t1']

                        neigh = Variable(neighbors.cuda(), requires_grad=False)
                        neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                        neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)

                        label_OHE_2, xhat = second_model.model(neigh[diffidx], neigh_z[diffidx], neigh_y[diffidx])
                        label_OHE[diffidx] = label_OHE_2

                elif self.uncertainty:
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
                # self.Zs.append(Z_dec)
                self.xhats.append(xhat.detach())
                # self.Z_fs.append(Z_f.detach())
                self.maxindxs.append(maxindx.detach().data)


            labeled = torch.cat(label_OHEs, 0)
            labeled = labeled.data.cpu().numpy()
            maxindices = torch.cat(self.maxindxs, 0).cpu().numpy()

            # labeled_list.append(labeled)

            # labeled = np.argmax(labeled, axis=1)
            if dataloader is None:
                size = self.data[i].dataset.dataOrigShape[:3]
                origindices = self.data[i].dataset.indices
            else:
                size = dataloader.dataset.dataset.dataOrigShape[:3]
                origindices = dataloader.dataset.dataset.indices

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

            recon = nib.Nifti1Image(Y, affine=affine[i])
            nib.save(recon, filename=intimgname[i])

            recon = nib.Nifti1Image(X, affine=affine[i])
            nib.save(recon, filename=intoutname[i])

            labeled_list.append(L.reshape(np.prod(size3d[:3]), 3))

        #     X_list.append(X)
        #     self.xhat_list.append(self.xhats)
        #
        #
        # self.set_mode('train')

        # return X_list
        return labeled_list



