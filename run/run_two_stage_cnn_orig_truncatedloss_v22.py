import numpy as np
# import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from Dataset import MRDataSet_v22, MRDataSet_mult_v22, MRDataSet_test_mult_v22, MRDataSet_2planes_v22, \
    MRDataSet_2planes_mult_v22, MRDataSet_test_2planes_mult_v22, MRDataSet_test_2planes_v22
import torch
from torch.utils.data import ConcatDataset
# from sklearn.neighbors import BallTree, KDTree
from torch.nn import DataParallel
from models import two_stage_cnn_uncertainty, two_stage_cnn, two_stage_cnn_dynSize, two_stage_cnn_2planes, \
    two_stage_cnn_2planes_with_uncertainty
import nibabel as nib
# import itertools
# import h5py
from Dataset import MRDataSet_test_v22
from sklearn.model_selection import train_test_split
# import torch.nn.functional as F
import time
from Dataset.multiEpochLoader import MultiEpochsDataLoader

processes = []

from truncatedloss import TruncatedLoss

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


class Solver(object):
    def __init__(self, obj, crossfold=4, valobj=None, epoch=30, batch_size=100000, lr=2e-4, f_dim =5, pad =5, lossThresh = 0.2,
                 beta=0.25, labels=3, shuffle=True, valuncertfn = None, numslices =(10,0,0),
                 channels=1, uncertainty = False, uncertfn = None, slices = False, axes = (True,True,True),
                 channels2 = 3, textfn='/data/avgloss.txt', suffix = '', dataset_portion = 0.5, num_workers = 6,
                 dynSize = False, width=21, height=21, twoplane = None
                 ):
        self.num_workers = num_workers
        self.obj = obj
        self.crossfold = crossfold
        self.valNum = int(np.ceil(len(obj)/crossfold))
        self.trainNum = len(obj) - self.valNum
        self.count = np.zeros(len(self.obj), dtype=np.int16)
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
        self.dynSize = dynSize
        self.Width = width
        self.Height = height
        self.twoplane = twoplane
        if self.uncertainty:
            if self.twoplane:
                self.model = two_stage_cnn_2planes_with_uncertainty.model_2input_mirrored(
                    f_dim=self.f_dim,
                    pad=self.pad,
                    labels=self.labels,
                    channels=self.channels)
            else:
                self.model = two_stage_cnn_uncertainty.model_2input_mirrored(
                    f_dim=self.f_dim,
                    pad=self.pad,
                    labels=self.labels,
                    channels=self.channels,
                    channels2=self.channels2)
        elif self.dynSize:
            self.model = two_stage_cnn_dynSize.model_2input_mirrored(
                f_dim=self.f_dim,
                width=self.Width,
                height=self.Height,
                labels=self.labels,
                channels=self.channels,
                )
        elif self.twoplane:
            self.model = two_stage_cnn_2planes.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                labels=self.labels,
                channels=self.channels
            )
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
        self.valL1_Loss = nn.L1Loss().cuda()
        self.MSE_Loss = nn.MSELoss().cuda()
        self.CE_Loss = nn.BCELoss().cuda()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def prep_data(self):
        flip = np.max(self.count) - self.count + 1
        norm = np.sum(flip)
        denom = 1
        shift = 0
        allidxs = np.arange(len(self.obj))
        probmap = ((flip + shift) / (norm + shift)) / denom
        valIdxs = np.random.choice(range(0, len(self.obj)), self.valNum, p=probmap)
        trainIdxs = allidxs[~np.isin(allidxs, valIdxs)]
        self.count[valIdxs] +=1
        self.data = []
        if not self.uncertainty:
            if self.twoplane:
                for i in trainIdxs:
                    tmpdata = MRDataSet_2planes_v22.MRDataSet(file=self.obj[i],
                                                              transform=transforms.Compose(
                                                                  [MRDataSet_2planes_v22.ToTensor()]),
                                                              render=True, slices=self.slices, axes=self.axes,
                                                              numslices=self.numslices, planes=self.twoplane, pad=self.pad)
                    self.data.append(tmpdata)
            else:
                for i in trainIdxs:
                    tmpdata = MRDataSet_v22.MRDataSet(file=self.obj[i],
                                                      transform=transforms.Compose([MRDataSet_v22.ToTensor()]),
                                                      render=True, slices=self.slices, axes=self.axes,
                                                      numslices=self.numslices, pad=self.pad)
                    self.data.append(tmpdata)
        elif self.uncertainty:
            if self.twoplane:
                for i in trainIdxs:
                    tmpdata = MRDataSet_2planes_mult_v22.MRDataSet(file=self.obj[i], file2=self.uncertfn[i],
                                                        transform=transforms.Compose([MRDataSet_2planes_mult_v22.ToTensor()]),
                                                      render=True, slices=self.slices, axes=self.axes,
                                                      numslices=self.numslices, planes=self.twoplane, pad=self.pad)
                    self.data.append(tmpdata)
            else:
                for i in trainIdxs:
                    tmpdata = MRDataSet_mult_v22.MRDataSet(file=self.obj[i], file2=self.uncertfn[i],
                                                        transform=transforms.Compose([MRDataSet_mult_v22.ToTensor()]),
                                                      render=True, slices=self.slices, axes=self.axes,
                                                      numslices=self.numslices, pad=self.pad)
                    self.data.append(tmpdata)
        del tmpdata
        self.allTrainDataNum = ConcatDataset(self.data).cumulative_sizes[-1]
        self.valdata = []
        if not self.uncertainty:
            if self.twoplane:
                for i in valIdxs:
                    tmpdata = MRDataSet_2planes_v22.MRDataSet(file=self.obj[i],
                                                              transform=transforms.Compose(
                                                                  [MRDataSet_2planes_v22.ToTensor()]),
                                                              render=True, slices=self.slices, axes=self.axes,
                                                              numslices=self.numslices, planes=self.twoplane, pad=self.pad)
                    self.valdata.append(tmpdata)
            else:
                for i in valIdxs:
                    tmpdata = MRDataSet_v22.MRDataSet(file=self.obj[i],
                                                      transform=transforms.Compose([MRDataSet_v22.ToTensor()]),
                                                      render=True, slices=self.slices, axes=self.axes,
                                                      numslices=self.numslices, pad=self.pad)
                    self.valdata.append(tmpdata)
        elif self.uncertainty:
            if self.twoplane:
                for i in valIdxs:
                    tmpdata = MRDataSet_2planes_mult_v22.MRDataSet(file=self.obj[i], file2=self.uncertfn[i],
                                                                   transform=transforms.Compose(
                                                                       [MRDataSet_2planes_mult_v22.ToTensor()]),
                                                                   render=True, slices=self.slices, axes=self.axes,
                                                                   numslices=self.numslices, planes=self.twoplane, pad=self.pad)
                    self.valdata.append(tmpdata)
            else:
                for i in valIdxs:
                    tmpdata = MRDataSet_mult_v22.MRDataSet(file=self.obj[i], file2=self.uncertfn[i],
                                                        transform=transforms.Compose([MRDataSet_mult_v22.ToTensor()]),
                                                  render=True, slices=self.slices, axes=self.axes,
                                                  numslices=self.numslices, pad=self.pad)
                    self.valdata.append(tmpdata)
        del tmpdata
        self.allValDataNum = ConcatDataset(self.valdata).cumulative_sizes[-1]
        self.valcriterion = TruncatedLoss(trainset_size=self.allValDataNum).cuda()
        self.criterion = TruncatedLoss(trainset_size=self.allTrainDataNum).cuda()

    def reset_model(self):
        if self.uncertainty:
            if self.twoplane:
                self.model = two_stage_cnn_2planes_with_uncertainty.model_2input_mirrored(
                    f_dim=self.f_dim,
                    pad=self.pad,
                    labels=self.labels,
                    channels=self.channels)
            else:
                self.model = two_stage_cnn_uncertainty.model_2input_mirrored(
                    f_dim=self.f_dim,
                    pad=self.pad,
                    labels=self.labels,
                    channels=self.channels,
                    channels2=self.channels2)
        elif not self.uncertainty:
            if self.twoplane:
                self.model = two_stage_cnn_2planes.model_2input_mirrored(
                    f_dim=self.f_dim,
                    pad=self.pad,
                    labels=self.labels,
                    channels=self.channels
                )
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
        # self.criterion = TruncatedLoss(trainset_size=self.traintotalAmount).cuda()
        # self.valcriterion = TruncatedLoss(trainset_size=self.valtotalAmount).cuda()

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
                self.prep_data()
                subdatas = []
                for dnum in range(0,len(self.data)):
                    self.trainidxs0 = np.where(self.data[dnum].labels == 0)[0]
                    self.trainidxs1 = np.where(self.data[dnum].labels == 1)[0]
                    total1 = len(self.trainidxs1)
                    sampled0 = np.random.choice(np.arange(self.trainidxs0.size), total1)
                    sampledidxs0 = self.trainidxs0[sampled0]
                    sublabels0 = self.data[dnum].labels[sampledidxs0]
                    reconlabels = np.concatenate([sublabels0, self.data[dnum].labels[self.trainidxs1]])
                    X_train, _, y_train, _ = train_test_split(np.concatenate([self.trainidxs1, sampledidxs0]), reconlabels,
                                                              train_size=self.dataset_portion,
                                                              stratify=reconlabels)
                    assert np.abs(np.count_nonzero(y_train) - np.count_nonzero(y_train == 0)) < 3
                    subdata = torch.utils.data.Subset(self.data[dnum], X_train)
                    subdatas.append(subdata)
                traindata = ConcatDataset(subdatas)
                self.dataloader = MultiEpochsDataLoader(traindata, batch_size=self.batch_size, shuffle=self.shuffle,
                                             num_workers=self.num_workers, drop_last=False)
                epochstart = time.time()
                for idx, (sample, indices, orig_indices, _) in enumerate(self.dataloader):
                    elapsed = time.time() - epochstart
                    print('\nBatch {0}'.format(idx))
                    label_losses = []
                    train_loss = 0
                    correct = 0
                    total = 0
                    if self.uncertainty:
                        if not self.twoplane:
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
                            neighbors1 = sample['neighbors1']
                            neighbors2 = sample['neighbors2']
                            uncert1 = sample['uncert1']
                            uncert2 = sample['uncert2']
                            ylabel = sample['label']

                            neigh1 = Variable(neighbors1.cuda(), requires_grad=False)
                            neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                            y = Variable(ylabel.cuda(), requires_grad=False)
                            label_OHE, xhat, maxindx = self.model(neigh1, neigh2, uncert1, uncert2)

                    elif self.twoplane:
                        neighbors1 = sample['neighbors1']
                        neighbors2 = sample['neighbors2']
                        ylabel = sample['label']

                        neigh1 = Variable(neighbors1.cuda(), requires_grad=False)
                        neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                        y = Variable(ylabel.cuda(), requires_grad=False)

                        label_OHE, xhat, maxindx = self.model(neigh1, neigh2)

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

                    # truncloss =  self.criterion(label_OHE, y) + 0.5*self.L1_Loss(label_OHE, F.one_hot(y, self.labels))
                    truncloss = self.criterion(label_OHE, y, indices)
                    self.optimizer.zero_grad()
                    truncloss.backward()
                    self.optimizer.step()

                    label_losses.append(truncloss.detach().data)

                    train_loss += truncloss.item()
                    _, predicted = torch.max(label_OHE.data, 1)
                    total += y.size(0)
                    correct += predicted.eq(ylabel.cuda().data).cpu().sum()
                    correct = correct.item()
                    label_losses_cat = torch.stack(label_losses, 0).mean()
                    # txt = 'Average loss per batch: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (idx + 1), 100. * correct / total, correct, total)
                    txt = 'Average loss per batch: %.3f | Acc: %.3f%% (%d/%d)' % (
                    label_losses_cat.data, 100. * correct / total, correct, total)
                    print(txt)
                    with open(self.textfn, 'a+') as f:
                        f.write("{0}\n".format(label_losses_cat))

                ## TODO: figure out what's happening with the loss
                # if e ==2 and not self.uncertainty:
                #     label_losses_cat = torch.stack(label_losses, 0).mean()
                #     if label_losses_cat > self.lossThresh:
                #         restart = True
                #         self.reset_model()
                #         break

                # if self.valobj:

                ### Validation portion ####
                self.set_mode('eval')
                print("\nValidation:")
                subdatas = []
                for dnum in range(0, len(self.valdata)):
                    self.validxs0 = np.where(self.valdata[dnum].labels == 0)[0]
                    self.validxs1 = np.where(self.valdata[dnum].labels == 1)[0]
                    total1 = len(self.validxs1)
                    sampled0 = np.random.choice(np.arange(self.validxs0.size), total1)
                    sampledidxs0 = self.validxs0[sampled0]
                    sublabels0 = self.valdata[dnum].labels[sampledidxs0]
                    reconlabels = np.concatenate([sublabels0, self.valdata[dnum].labels[self.validxs1]])

                    X_train, _, y_train, _ = train_test_split(np.concatenate([self.validxs1, sampledidxs0]),
                                                              reconlabels,
                                                              train_size=self.dataset_portion,
                                                              stratify=reconlabels)
                    assert np.abs(np.count_nonzero(y_train) - np.count_nonzero(y_train == 0)) < 3
                    subdata = torch.utils.data.Subset(self.valdata[dnum], X_train)
                    subdatas.append(subdata)
                valdata = ConcatDataset(subdatas)
                self.valdataloader = MultiEpochsDataLoader(valdata, batch_size=self.batch_size,
                                                shuffle=self.shuffle, num_workers=self.num_workers, drop_last=False)

                with torch.no_grad():
                    for idx, (sample, indices, orig_indices, _) in enumerate(self.valdataloader):
                        label_losses = []
                        val_train_loss = 0
                        val_correct = 0
                        val_total = 0
                        if self.uncertainty:
                            if not self.twoplane:
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
                                neighbors1 = sample['neighbors1']
                                neighbors2 = sample['neighbors2']
                                uncert1 = sample['uncert1']
                                uncert2 = sample['uncert2']
                                ylabel = sample['label']

                                neigh1 = Variable(neighbors1.cuda(), requires_grad=False)
                                neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                                y = Variable(ylabel.cuda(), requires_grad=False)
                                # y = F.one_hot(y, num_classes=2)

                                label_OHE, xhat, maxindx = self.model(neigh1, neigh2, uncert1, uncert2)

                        elif self.twoplane:
                            neighbors1 = sample['neighbors1']
                            neighbors2 = sample['neighbors2']
                            ylabel = sample['label']

                            neigh1 = Variable(neighbors1.cuda(), requires_grad=False)
                            neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                            y = Variable(ylabel.cuda(), requires_grad=False)

                            label_OHE, xhat, maxindx = self.model(neigh1, neigh2)

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

                        truncloss = self.valcriterion(label_OHE, y, indices) # + self.valL1_Loss(label_OHE, F.one_hot(y, self.labels))
                        label_losses.append(truncloss.detach().data)

                        val_train_loss += truncloss.item()
                        _, predicted = torch.max(label_OHE.data, 1)
                        val_total += y.size(0)
                        val_correct += predicted.eq(ylabel.cuda().data).cpu().sum()
                        val_correct = val_correct.item()

                        label_losses_cat = torch.stack(label_losses, 0).mean()
                        txt = '[Validation] Average loss per batch: %.3f | Acc: %.3f%% (%d/%d)' % (
                            label_losses_cat.data, 100. * val_correct / val_total, val_correct, val_total)
                        print(txt)
                        # txt = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        # val_train_loss / (idx + 1), 100. * val_correct / val_total, val_correct, val_total)
                        # print(txt)
                        with open(self.textfn.split('.')[0]+'_val.txt', 'a+') as f:
                            f.write("{0}\n".format(label_losses_cat))

                    print("Validation end.\n")
                elapsed = time.time() - epochstart
                print('\n>> Epoch {0} total time:'.format(e + 1), elapsed / 60)
                ## TODO: implement termination criteria

    def test(self, intimgname, intoutname, affine, dataset=None, dataset2=None, batchsize =1000,
             slices = False, axes = (True, True, True), numslices = (10,0,0), num_workers = 6, dontWriteInter=False):
        import torch.multiprocessing
        torch.multiprocessing.set_sharing_strategy('file_system')
        self.set_mode('eval')
        self.model.eval()
        self.xhat_list = []
        self.xhats = []
        self.maxindxs = []
        label_OHEs = []
        self.maxindxs_mod = []
        origindices = []

        if not self.uncertainty:
            if not self.twoplane:
                data = MRDataSet_test_v22.MRDataSet(dataset,transform=transforms.Compose([MRDataSet_test_v22.ToTensor()]),
                                                              render=True,
                                                           slices=slices, axes=axes,
                                                              numslices=numslices, pad=self.pad
                                                           )
            elif self.twoplane:
                data = MRDataSet_test_2planes_v22.MRDataSet(dataset,
                                                   transform=transforms.Compose(
                                                       [MRDataSet_test_2planes_v22.ToTensor()]),
                                                   render=True, slices=slices, axes=self.axes,
                                                   numslices=numslices, planes=self.twoplane, pad=self.pad)
        else:
            if not self.twoplane:
                data = MRDataSet_test_mult_v22.MRDataSet(dataset, dataset2,
                                                                    transform=transforms.Compose(
                                                                        [MRDataSet_test_mult_v22.ToTensor()]),
                                                                    render=True, slices=slices, axes=axes,
                                                                    numslices=numslices, pad=self.pad)
            else:
                data = MRDataSet_test_2planes_mult_v22.MRDataSet(dataset, dataset2,
                                                         transform=transforms.Compose(
                                                             [MRDataSet_test_2planes_mult_v22.ToTensor()]),
                                                         render=True, slices=slices, axes=axes,
                                                         numslices=numslices, planes=self.twoplane, pad=self.pad)
        ind_dataloader = DataLoader(data, batch_size=batchsize, shuffle=False,
                                            num_workers=num_workers, drop_last=False)
        size = dataset.origsize[:3]
        padsize = dataset.data.shape
        padsize3 = padsize[:3]
        for idx, (sample, indices, orig_indices) in enumerate(ind_dataloader):

            if self.uncertainty:
                if not self.twoplane:
                    neighbors = sample['neighbors']
                    neighbors_z = sample['neighbors_z']
                    neighbors_y = sample['neighbors_y']

                    neighbors2 = sample['neighbors2']
                    neighbors2_z = sample['neighbors2_z']
                    neighbors2_y = sample['neighbors2_y']

                    neigh = Variable(neighbors.cuda(), requires_grad=False)
                    neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                    neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                    neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                    neigh2_z = Variable(neighbors2_z.cuda(), requires_grad=False)
                    neigh2_y = Variable(neighbors2_y.cuda(), requires_grad=False)

                    label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y, neigh2, neigh2_z, neigh2_y)
                else:
                    neighbors1 = sample['neighbors1']
                    neighbors2 = sample['neighbors2']
                    uncert1 = sample['uncert1']
                    uncert2 = sample['uncert2']

                    neigh1 = Variable(neighbors1.cuda(), requires_grad=False)
                    neigh2 = Variable(neighbors2.cuda(), requires_grad=False)

                    label_OHE, xhat, maxindx = self.model(neigh1, neigh2, uncert1, uncert2)

            elif self.twoplane:
                neighbors1 = sample['neighbors1']
                neighbors2 = sample['neighbors2']

                neigh1 = Variable(neighbors1.cuda(), requires_grad=False)
                neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                label_OHE, xhat, maxindx = self.model(neigh1, neigh2)
            else:
                neighbors = sample['neighbors']
                neighbors_z = sample['neighbors_z']
                neighbors_y = sample['neighbors_y']

                neigh = Variable(neighbors.cuda(), requires_grad=False)
                neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y)

            origindices.append(orig_indices)

            label_OHEs.append(label_OHE.detach().cpu().numpy())

            self.xhats.append(xhat.detach().cpu().numpy())
            self.maxindxs.append(maxindx.detach().cpu().numpy())

        del ind_dataloader
        origindices = np.concatenate(origindices)
        labeled = np.concatenate(label_OHEs, 0)
        del label_OHEs
        maxindices = np.concatenate(self.maxindxs, 0) + 1
        del self.maxindxs

        self.xhats = np.vstack(self.xhats)

        size4d = np.concatenate((padsize3, [self.f_dim]))
        size3d = np.concatenate((padsize3, [self.labels]))

        if not dontWriteInter:
            Y = np.zeros(size4d)
            Y.reshape((np.prod(padsize3), self.f_dim))[origindices] = self.xhats
            # remove padding
            Y = Y[self.pad:-self.pad, self.pad:-self.pad,self.pad:-self.pad]
            recon = nib.Nifti1Image(Y, affine=affine)
            del Y #, values
            nib.save(recon, filename=intimgname)
            del recon

        X = np.zeros(padsize3)
        X.ravel()[origindices] = maxindices
        X = X[self.pad:-self.pad, self.pad:-self.pad, self.pad:-self.pad]
        recon = nib.Nifti1Image(X.astype(np.int16), affine=affine)
        del X
        nib.save(recon, filename=intoutname)
        del recon

        L = np.zeros(size3d)
        L.reshape((np.prod(padsize3), self.labels))[origindices] = labeled
        L = L[self.pad:-self.pad, self.pad:-self.pad, self.pad:-self.pad]
        L = L.reshape(np.prod(size), 2)

        del orig_indices

        return L



