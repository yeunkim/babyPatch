import numpy as np
# import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from Dataset import MRDataSet2_noupsample, MRDataSet2_mult_dataset
import torch
from torch.utils.data import ConcatDataset
# from sklearn.neighbors import BallTree, KDTree
from torch.nn import DataParallel
from models import two_stage_cnn_uncertainty, two_stage_cnn
import nibabel as nib
import itertools
import pickle

processes = []

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
                 start_prune = 2, suffix = '', spherecoord=False, lossThresh = 0, initlabels = None,
                 valinitlabels = None):
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
        self.lossThresh = lossThresh
        if initlabels:
            self.initlabels = []
            for i in range(len(initlabels)):
                dat = nib.load(initlabels[i]).get_fdata()
                self.initlabels.append(dat)
        if valinitlabels:
            self.initlabelsVal = []
            for i in range(len(valinitlabels)):
                dat = nib.load(valinitlabels[i]).get_fdata()
                self.initlabelsVal.append(dat)

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
        self.CE_Loss_val = nn.CrossEntropyLoss().cuda()

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
                                                          multiinput=multiinput, coords=self.coords, )
                self.data.append(tmpdata)

        elif self.uncertainty:
            self.data = []
            for i in np.arange(len(self.obj)):
                tmpdata = MRDataSet2_mult_dataset.MRDataSet(pkl_file=self.obj[i], pkl_file2=self.uncertfn[i],
                                                            transform=transforms.Compose([
                                                       MRDataSet2_mult_dataset.ToTensor(multiinput=multiinput,
                                                                                        coords=self.coords,
                                                                                        spherecoord=self.spherecoord)
                                                   ]), miscidxs=self.miscidx, spherecoord=self.spherecoord,
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

    def gkern(self, l=11, sig=1.):
        """
        creates gaussian kernel with side length `l` and a sigma of `sig`
        """
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
        kernel = np.outer(gauss, gauss)
        return kernel / np.sum(kernel)


    # TODO: compute variance of the label/texture
    def compute_label_variance(self, labelStack, Wkernel):
        sd = (labelStack - labelStack[:,0,5,5][:,None,None,None]) # element-wise square
        var = sd*sd
        # TODO: gaussian weighted sum on one-shifted label (0 is a non-labeled area)
        var[labelStack == 0 ] = 0
        weightedSum = torch.sum(torch.sum(var*Wkernel, 2),2)
        return torch.mean(weightedSum)

    def find_UL_LR(self, centers, shape, width):
        ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]
        limits = np.zeros((len(centers), 8))
        for i in range(len(centers)):
            cx, cy, cz = np.unravel_index(centers[i], shape[i][:3])
            # limits = ravel((width,width,width),shape[i])
            # ULs[i] = centers[i] - limits
            # LRs[i] = centers[i] + limits
            ii = 0
            for x in [-1, 1]:
                for j in [-1,1]:
                    for k in [-1,1]:
                        tmp = (cx+(x*width), cy+(j*width), cz+(k*width))
                        limits[i][ii] = ravel(tmp, shape[i])
                        ii +=1
            # ULs[i] =
            # LRs[i] = centers[i] + limits
        return limits

    def train(self, **kwargs):
        self.set_mode('train')
        if 'epoch' in kwargs:
            self.epoch = kwargs['epoch']
        if 'lr' in kwargs:
            self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['lr'], betas=(0.5, 0.999))

        terminate = 0
        width = (2 * self.pad) + 1
        prev_acc = 0
        val_prev_acc = 0
        numTraindata = len(self.obj)
        numTraindataVal = len(self.valobj)
        ROIwidth = int(25)
        totalWidth = int((2 * ROIwidth) +1)

        gaussW = self.gkern(width, sig=1)
        Wkernel = torch.tensor(gaussW).cuda()
        tmpindarr = np.expand_dims(np.arange(-1*self.pad, self.pad+1),1)
        tmpz = (((totalWidth**2) * np.tile(tmpindarr,width)) + tmpindarr.transpose()).reshape(-1)
        tmps = np.empty((3,tmpz.size))
        tmps[2] = tmpz
        tmps[0] = (((totalWidth ** 2) * np.tile(tmpindarr, width)) + (totalWidth * tmpindarr)).reshape(-1)
        tmps[1] = ((totalWidth * tmpindarr) + tmpindarr.transpose()).reshape(-1)

        restart = True
        traindata = ConcatDataset(self.data)
        valdata = ConcatDataset(self.valdata)
        # TODO: load in initialized labels here, offset by 1, 0 where it's not labeled
        # initialize label nd array for indexing (pad 0 if smaller than max label size)
        biggestLabel = np.argmax(np.asarray([ self.initlabels[ii].size for ii in range(numTraindata)]))
        sx, sy, sz = self.initlabels[biggestLabel].shape
        tdimsize = sx*sy*sz
        initlabelsIdxs = np.zeros((numTraindata, sx,sy,sz)).astype(np.int16)
        biggestLabelVal = np.argmax(np.asarray([self.initlabelsVal[ii].size for ii in range(numTraindataVal)]))
        vsx, vsy, vsz = self.initlabelsVal[biggestLabelVal].shape
        vtdimsize = vsx*vsy*vsz
        initlabelsIdxsVal = np.zeros((numTraindataVal,vsx,vsy,vsz)).astype(np.int16)

        traindata_indices = []
        traindata_shapes =[]
        index_mapping = []
        valdata_indices = []
        valdata_shapes = []
        valindex_mapping = []
        startI = 0
        for i in range(numTraindata):
            data= pickle.load(open(self.obj[i], 'rb'))
            traindata_indices.append(data.indices)
            traindata_shapes.append(data.dataOrigShape)
                # traindata_indices.append(data['indices'])
                # traindata_shapes.append(data['origsize'])
            # with h5py.File(self.obj[i], 'r') as data:
            mapping = np.empty(traindata_shapes[i])
            mapping.fill(np.int16(-1))
            mapping = mapping.ravel()
            mapping[traindata_indices[i]] = np.arange(startI, startI + len(traindata_indices[i]))
            index_mapping.append(mapping)
            startI = len(traindata_indices[i])
        startI = 0
        for i in range(numTraindataVal):
            data= pickle.load(open(self.valobj[i], 'rb'))
            valdata_indices.append(data.indices)
            valdata_shapes.append(data.dataOrigShape)
            # with h5py.File(self.valobj[i], 'r') as data:
            mapping = np.empty(valdata_shapes[i])
            mapping.fill(np.int16(-1))
            mapping= mapping.ravel()
            mapping[valdata_indices[i]] = np.arange(startI, startI + len(valdata_indices[i]))
            valindex_mapping.append(mapping)
            startI = len(valdata_indices[i])

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

                initlabelsIdxs = initlabelsIdxs.reshape((numTraindata, sx,sy,sz))

                # Select random center voxels for ROI region
                centerVoxels = [np.random.choice(traindata_indices[c],1) for c in range(numTraindata)]
                # Find upper left and lower right boundary points
                limits = self.find_UL_LR(centerVoxels, traindata_shapes, ROIwidth)
                # Segment out the appropriate regions in the initialized labels
                initlabels = np.empty((numTraindata, totalWidth, totalWidth, totalWidth))
                for ii in range(numTraindata):
                    cx, cy, cz = np.unravel_index(centerVoxels[ii], traindata_shapes[ii][:3])
                    cx, cy, cz = cx[0], cy[0], cz[0]
                    length = totalWidth**3
                    initlabels[ii] = self.initlabels[ii][cx-ROIwidth:cx+ROIwidth+1,cy-ROIwidth:cy+ROIwidth+1,
                                     cz-ROIwidth:cz+ROIwidth+1]
                    initlabelsIdxs[ii][cx-ROIwidth:cx+ROIwidth+1,cy-ROIwidth:cy+ROIwidth+1,
                                     cz-ROIwidth:cz+ROIwidth+1] = np.arange(0, length).reshape((totalWidth, totalWidth,
                                                                                                totalWidth))
                initlabelsIdxs = initlabelsIdxs.reshape(numTraindata,tdimsize)
                initlabels = initlabels.reshape(numTraindata,totalWidth**3)
                # Select out voxels that were labeled in the training dataset
                subsetIdxs = np.concatenate(
                    [index_mapping[m][traindata_indices[m][(traindata_indices[m] > limits[m,0]) &
                                                           (traindata_indices[m] > limits[m, 1]) &
                                                           (traindata_indices[m] > limits[m, 2]) &
                                                           (traindata_indices[m] > limits[m, 3]) &
                                                           (traindata_indices[m] < limits[m, 4]) &
                                                            (traindata_indices[m] < limits[m, 5]) &
                                                            (traindata_indices[m] < limits[m, 6]) &
                                                            (traindata_indices[m] < limits[m, 7])]]
                     for m in range(numTraindata)])
                # Sanity check: there shouldn't be any unlabeled voxels
                assert np.any(subsetIdxs != -1)
                # Subset data only in the ROI region
                subsetdata = torch.utils.data.Subset(traindata, subsetIdxs.astype(np.int))
                subsetdataloader = DataLoader(subsetdata, batch_size=self.batch_size, shuffle=True, drop_last=False)

                # TODO: initialize a small label area with zeros
                # TODO: iteration 0 - run through ALL voxels and then compute label variance: 1) subtract everything by 1, 2) (x - x_c)^2, 3) weighted sum using gaussian weighting
                # for now use initialized label
                # flatlabel = np.zeros((len(traindata_indices),totalWidth**3)).astype(np.int16)

                # TODO: update as it goes along to patch - variance should be reduced

                for idx, (sample, indices, orig_indices, datasetNums) in enumerate(subsetdataloader):
                    actual_batch_num = len(indices)
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
                    # TODO: put new labels into the label file
                    labelvar = 0
                    labelcoords = initlabelsIdxs[tuple((datasetNums, indices.detach().cpu().numpy()))]
                    initlabels[tuple((datasetNums, labelcoords))] = maxindx.detach().cpu().numpy()
                    for dim in range(3):
                        tmp = np.tile(tmps[dim], (actual_batch_num, 1))
                        new_label_coords_xz = (np.expand_dims(labelcoords, 1) + tmp).astype(np.int16).ravel()
                        DataNumInd = np.repeat(datasetNums, width ** 2, axis=0)
                        labeltmp = torch.tensor(initlabels[tuple((DataNumInd,
                                                              new_label_coords_xz))].reshape(-1, 1, width,
                                                                                          width).astype(
                            'float32')).cuda()
                        labelvar += self.compute_label_variance(labeltmp, Wkernel)
                    labelvar /= 3
                    # truncloss =  self.criterion(label_OHE, y, indices)  + labelvar
                    truncloss = self.CE_Loss(label_OHE, y) + labelvar
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

                # if e ==2 and not self.uncertainty:
                #     if label_losses_cat > self.lossThresh:
                #         restart = True
                #         self.reset_model()
                #         break

                if self.valobj:
                    self.set_mode('eval')
                    print("Validation:")
                    initlabelsIdxsVal = initlabelsIdxsVal.reshape((numTraindataVal, vsx, vsy, vsz))
                    centerVoxels = [np.random.choice(valdata_indices[c], 1) for c in range(numTraindataVal)]
                    # Find upper left and lower right boundary points
                    limits = self.find_UL_LR(centerVoxels, valdata_shapes, ROIwidth)
                    # Segment out the appropriate regions in the initialized labels
                    initlabelsVal = np.empty((numTraindataVal, totalWidth, totalWidth, totalWidth))
                    for ii in range(numTraindataVal):
                        cx, cy, cz = np.unravel_index(centerVoxels[ii], valdata_shapes[ii][:3])
                        cx, cy, cz = cx[0], cy[0], cz[0]
                        length = totalWidth ** 3
                        initlabelsVal[ii] = self.initlabelsVal[ii][cx - ROIwidth:cx + ROIwidth + 1,
                                         cy - ROIwidth:cy + ROIwidth + 1,
                                         cz - ROIwidth:cz + ROIwidth + 1]
                        initlabelsIdxsVal[ii][cx - ROIwidth:cx + ROIwidth + 1, cy - ROIwidth:cy + ROIwidth + 1,
                        cz - ROIwidth:cz + ROIwidth + 1] = np.arange(0, length).reshape((totalWidth, totalWidth,
                                                                                         totalWidth))
                    initlabelsIdxsVal = initlabelsIdxsVal.reshape(numTraindataVal, vtdimsize)
                    initlabelsVal = initlabelsVal.reshape(numTraindataVal, totalWidth ** 3)
                    # Select out voxels that were labeled in the training dataset
                    subsetIdxs = np.concatenate(
                        [valindex_mapping[m][valdata_indices[m][(valdata_indices[m] > limits[m, 0]) &
                                                               (valdata_indices[m] > limits[m, 1]) &
                                                               (valdata_indices[m] > limits[m, 2]) &
                                                               (valdata_indices[m] > limits[m, 3]) &
                                                               (valdata_indices[m] < limits[m, 4]) &
                                                               (valdata_indices[m] < limits[m, 5]) &
                                                               (valdata_indices[m] < limits[m, 6]) &
                                                               (valdata_indices[m] < limits[m, 7])]]
                         for m in range(numTraindataVal)])
                    # Sanity check: there shouldn't be any unlabeled voxels
                    assert np.any(subsetIdxs != -1)
                    # Subset data only in the ROI region
                    subsetdata = torch.utils.data.Subset(valdata, subsetIdxs.astype(np.int16))
                    subsetdataloader = DataLoader(subsetdata, batch_size=self.batch_size, shuffle=True, drop_last=False)
                    with torch.no_grad():
                        for idx, (sample, indices, orig_indices, datasetNums) in enumerate(subsetdataloader):
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

                            labelvar = 0
                            labelcoords = initlabelsIdxs[tuple((datasetNums, indices.detach().cpu().numpy()))]
                            initlabels[tuple((datasetNums, labelcoords))] = maxindx.detach().cpu().numpy()
                            actual_batch_num = len(indices)
                            for dim in range(3):
                                tmp = np.tile(tmps[dim], (actual_batch_num, 1))
                                new_label_coords_xz = (np.expand_dims(labelcoords, 1) + tmp).astype(np.int16).ravel()
                                DataNumInd = np.repeat(datasetNums, width ** 2, axis=0)
                                labeltmp = torch.tensor(initlabelsVal[tuple((DataNumInd,
                                                                          new_label_coords_xz))].reshape(-1, 1, width,
                                                                                                         width).astype(
                                    'float32')).cuda()
                                labelvar += self.compute_label_variance(labeltmp, Wkernel)
                            labelvar /= 3
                            # truncloss = self.valcriterion(label_OHE, y, indices)
                            truncloss = self.CE_Loss_val(label_OHE, y) + labelvar
                            loss = truncloss + labelvar

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
                # if e > 2 and not self.uncertainty:
                #     if prev_acc > (correct/total)+delta:
                #         terminate +=1
                #     elif terminate >0 :
                #         terminate -= 1
                #     if val_prev_acc > (val_correct / val_total) + delta:
                #         terminate += 1
                #     elif terminate >0 :
                #         terminate -= 1
                #     prev_acc = correct / total
                #     val_prev_acc = val_correct / val_total
                #     if terminate > 2:
                #         break
                # else:
                #     val_prev_acc = val_correct / val_total
                #     prev_acc = correct / total

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
                    data = MRDataSet2_noupsample.MRDataSet(pkl_file=imgs[d],
                                                           transform=transforms.Compose([
                                                                  MRDataSet2_noupsample.ToTensor(
                                                                      coords=self.coords,
                                                                      spherecoord=self.spherecoord)
                                                              ]), miscidxs=self.miscidx,
                                                           spherecoord=self.spherecoord,
                                                           coords=self.coords)
                    # data.append(tmpdata)

                elif self.uncertainty:
                    data = MRDataSet2_mult_dataset.MRDataSet(pkl_file=imgs[d], pkl_file2=uncertfn[d],
                                                             transform=transforms.Compose([
                                                                    MRDataSet2_mult_dataset.ToTensor(
                                                                        coords=self.coords,
                                                                        spherecoord=self.spherecoord)
                                                                ]), miscidxs=self.miscidx,
                                                             spherecoord=self.spherecoord, coords=self.coords)
                ind_dataloader = DataLoader(data, batch_size=batchsize, shuffle=False,
                                            num_workers=6, drop_last=False)
                size = data.dataset.dataOrigShape[:3]
                origindices = data.dataset.indices
                del data

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

                label_OHEs.append(label_OHE.detach())

                self.xhats.append(xhat.detach())
                self.maxindxs.append(maxindx.detach().data)

            del ind_dataloader

            labeled = torch.cat(label_OHEs, 0)
            del label_OHEs
            labeled = labeled.data.cpu().numpy()
            maxindices = torch.cat(self.maxindxs, 0).cpu().numpy()
            del self.maxindxs


            self.xhats = np.asarray(list(itertools.chain.from_iterable(self.xhats)))
            values = np.zeros([self.xhats.shape[0], self.f_dim])
            for B in np.arange(len(self.xhats)):
                values[B] = self.xhats[B].data.cpu().numpy()
            del self.xhats

            size4d = size + (self.f_dim,)
            size3d = size + (self.labels,)

            Y = np.zeros(size4d)
            for idx in np.arange(maxindices.shape[0]):
                idxs = np.unravel_index(origindices[idx], size)
                Y[idxs] = values[idx]
            recon = nib.Nifti1Image(Y, affine=affine[d])
            del Y, values
            nib.save(recon, filename=intimgname[d])
            del recon

            X = np.zeros(size)
            for idx in np.arange(maxindices.shape[0]):
                idxs = np.unravel_index(origindices[idx], size)
                X[idxs] = maxindices[idx] + 1
            recon = nib.Nifti1Image(X.astype(np.int16), affine=affine[d])
            recon.header.set_data_dtype(np.int16)
            del X

            nib.save(recon, filename=intoutname[d])
            del recon

            L = np.zeros(size3d)
            for idx in np.arange(maxindices.shape[0]):
                idxs = np.unravel_index(origindices[idx], size)
                L[idxs] = labeled[idx]

            labeled_list.append(L.reshape(np.prod(size3d[:3]), 3))

            del L
            del orig_indices

        return labeled_list



