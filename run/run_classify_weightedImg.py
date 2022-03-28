import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from Dataset import MRDataSet2_noupsample, MRDataSet2_mult_dataset
import torch
from torch.utils.data import ConcatDataset
from torch.nn import DataParallel
import nibabel as nib
import itertools
from truncatedloss import TruncatedLoss

processes = []

from models import two_stage_cnn, classify_weightedImg_with_uncertainty, ae_weightedImg_with_uncertainty
from datetime import datetime

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
                 beta=0.25, in_features=1, labels=3, shuffle=True, valuncertfn = None,
                 miscidx_val=None, params = None, channels=1, uncertainty = True, uncertfn = None,
                 iterative=False, textfn='/data/avgloss.txt', initmodel=None, suffix = '', ae=False):
        self.obj = obj
        self.valobj = valobj
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
        self.uncertainty = uncertainty
        self.uncertfn = uncertfn
        self.iterative= iterative
        self.textfn = textfn
        self.initmodel = initmodel
        self.suffix = suffix
        self.ae = ae
        self.valuncertfn = valuncertfn

        if self.uncertainty:
            self.model = two_stage_cnn.model_2input_mirrored(
                f_dim=self.f_dim,
                pad=self.pad,
                in_features=self.in_features,
                labels=self.labels,
                params=self.params,
                channels=self.channels)
            if self.ae:
                self.modelWImg = ae_weightedImg_with_uncertainty.model(
                    f1=20,
                    f2=40,
                    f3=20,
                    kw1=3,
                    kw2=4,
                    kw3=5,
                    f_dim=20,
                    pad=self.pad,
                    channels=self.channels,
                    g=30)
            else:
                self.modelWImg = classify_weightedImg_with_uncertainty.model(
                    f1=20,
                    f2=40,
                    f3=20,
                    kw1=1,
                    kw2=3,
                    kw3=5,
                    f_dim=self.f_dim,
                    pad=self.pad,
                    channels=self.channels)
            self.modelWImg = DataParallel(self.modelWImg)
            self.model = DataParallel(self.model)
            if self.initmodel:
                self.model.load_state_dict(torch.load(self.initmodel))

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
        self.model.eval()

        self.modelWImg = self.modelWImg.cuda()
        self.modelWImg.share_memory()

        # Criterions

        self.L1_Loss = nn.L1Loss().cuda()
        self.MSE_Loss = nn.MSELoss().cuda()

        self.CE_Loss = nn.CrossEntropyLoss().cuda()

        if not self.uncertainty:
            self.data =[]
            for i in np.arange(len(self.obj)):
                tmpdata = MRDataSet2_noupsample.MRDataSet(pkl_file=self.obj[i],
                                                          transform=transforms.Compose([
                                                       MRDataSet2_noupsample.ToTensor()
                                                   ]))
                self.data.append(tmpdata)

        elif self.uncertainty:
            self.data = []
            for i in np.arange(len(self.obj)):
                tmpdata = MRDataSet2_mult_dataset.MRDataSet(pkl_file=self.obj[i], pkl_file2=self.uncertfn[i],
                                                            transform=transforms.Compose([
                                                       MRDataSet2_mult_dataset.ToTensor()
                                                   ]), )
                self.data.append(tmpdata)


        self.dataloader = DataLoader(ConcatDataset(self.data), batch_size=self.batch_size, shuffle=self.shuffle,
                                     num_workers=2, drop_last=False)
        if self.valobj:
            self.valdata = []
            for i in np.arange(len(self.valobj)):
                tmpdata = MRDataSet2_mult_dataset.MRDataSet(pkl_file=self.valobj[i], pkl_file2=self.valuncertfn[i],
                                                            transform=transforms.Compose([
                                                                MRDataSet2_mult_dataset.ToTensor()
                                                            ]), )
                self.valdata.append(tmpdata)

            self.valdataloader = DataLoader(ConcatDataset(self.valdata), batch_size=self.batch_size, shuffle=self.shuffle,
                                         num_workers=5, drop_last=False)
        del tmpdata

        # Optimizer
        self.optimizerWImg = optim.Adam(self.modelWImg.parameters(), lr=self.lr, betas=(0.5, 0.999))
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.criterion = TruncatedLoss(trainset_size=len(ConcatDataset(self.data))).cuda()

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.modelWImg.train()
        elif mode == 'eval':
            self.modelWImg.eval()
        else:
            raise ('mode error. It should be either train or eval')

    def train(self, **kwargs):
        self.set_mode('train')
        # self.model.train()
        if 'epoch' in kwargs:
            self.epoch = kwargs['epoch']
        if 'lr' in kwargs:
            self.optimizer = optim.Adam(self.model.parameters(), lr=kwargs['lr'], betas=(0.5, 0.999))



        for e in range(self.epoch):
            print("Epoch {0}/{1}".format(e + 1, self.epoch))
            label_losses = []
            label_losses_mod = []

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

                    neigh = Variable(neighbors.cuda(), requires_grad=False)
                    neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                    neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                    y = Variable(ylabel.cuda(), requires_grad=False)
                    neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                    neigh2_z = Variable(neighbors2_z.cuda(), requires_grad=False)
                    neigh2_y = Variable(neighbors2_y.cuda(), requires_grad=False)

                    outputs = self.modelWImg(neigh, neigh_z, neigh_y, neigh2, neigh2_z, neigh2_y)

                    if self.ae:
                        (WImg, WImgz, WImgy, WImgU, WImgzU, WImgyU) = outputs
                        neighmod = WImg
                        neighzmod = WImgz
                        neighymod = WImgy
                        mseloss = self.MSE_Loss(WImg, neigh) + self.MSE_Loss(WImgz, neigh_z) + self.MSE_Loss(WImgy, neigh_y) \
                                  + self.MSE_Loss(WImgU, neigh2) + self.MSE_Loss(WImgzU, neigh2_z) + self.MSE_Loss(WImgyU, neigh2_y)
                    else:
                        (WImg, WImgz, WImgy) = outputs
                        neighmod = neigh + WImg
                        neighzmod = neigh_z + WImgz
                        neighymod = neigh_y + WImgy

                    label_OHE, xhat, maxindx = self.model(neighmod, neighzmod, neighymod)

                    # label_loss = self.CE_Loss(label_OHE, y)
                    #
                    # label_loss_reshaped = torch.repeat_interleave(label_loss,
                    #                                                   WImg.size()[0] *
                    #                                                   WImg.size()[-2] *
                    #                                                   WImg.size()[-1]).view(-1, 1,
                    #                                                                         WImg.size()[-2],
                    #                                                                         WImg.size()[-1])
                    # uncertainty_loss = torch.mean(neigh2, 2)
                    # uncertainty_loss_z = torch.mean(neigh2_z, 2)
                    # uncertainty_loss_y = torch.mean(neigh2_y, 2)
                    # total_loss = label_loss_reshaped # * uncertainty_loss
                    # total_loss_z = label_loss_reshaped # * uncertainty_loss_z
                    # total_loss_y = label_loss_reshaped # * uncertainty_loss_y
                    # torch.autograd.set_detect_anomaly(True)
                    # self.optimizerWImg.zero_grad()
                    # WImg.backward(total_loss, retain_graph=True)
                    # WImgz.backward(total_loss_z, retain_graph=True)
                    # WImgy.backward(total_loss_y, retain_graph=True)
                    # label_loss.backward()
                    # self.optimizerWImg.step()
                    # self.optimizer.zero_grad()

                    # self.optimizer.step()

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

                    # self.optimizer.zero_grad()
                    # label_loss.backward(retain_graph=True)
                    # total_loss = label_loss
                    # total_loss.backward(retain_graph=True)
                    # total_loss.backward()


                    # self.optimizer.step()

                    # diceloss = total_loss

                truncloss = self.criterion(label_OHE, y, indices)
                self.optimizerWImg.zero_grad()
                truncloss.backward(retain_graph=True)
                if self.ae:
                    mseloss.backward()

                self.optimizerWImg.step()

                label_losses.append(truncloss.detach().data)

                train_loss += truncloss.item()
                _, predicted = torch.max(label_OHE.data, 1)
                total += y.size(0)
                correct += predicted.eq(y.data).cpu().sum()
                correct = correct.item()

                if idx != 0 and idx % 10 == 0:
                    # AVG Losses
                    label_losses_mod_cat = 0
                    if self.iterative:
                        label_losses_mod_cat = torch.stack(label_losses_mod, 0).mean()
                    label_losses_cat = torch.stack(label_losses, 0).mean()
                    # total_losses_cat = torch.stack(total_losses, 0).mean()
                    # print('\n[{:02d}/{:d}] label_loss:{:.7f}'.format(
                    #     e+1,self.epoch, label_losses_cat))
                    txt = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    train_loss / (idx + 1), 100. * correct / total, correct, total)
                    print(txt)
                    with open(self.textfn, 'a+') as f:
                        f.write("{0}\t{1}\n".format(label_losses_cat, label_losses_mod_cat))

                del neigh, ylabel, sample, y

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

                            neigh = Variable(neighbors.cuda(), requires_grad=False)
                            neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                            neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                            y = Variable(ylabel.cuda(), requires_grad=False)
                            neigh2 = Variable(neighbors2.cuda(), requires_grad=False)
                            neigh2_z = Variable(neighbors2_z.cuda(), requires_grad=False)
                            neigh2_y = Variable(neighbors2_y.cuda(), requires_grad=False)

                            outputs = self.modelWImg(neigh, neigh_z, neigh_y, neigh2, neigh2_z, neigh2_y)

                            if self.ae:
                                (WImg, WImgz, WImgy, WImgU, WImgzU, WImgyU) = outputs
                                neighmod = WImg
                                neighzmod = WImgz
                                neighymod = WImgy
                            else:
                                (WImg, WImgz, WImgy) = outputs
                                neighmod = neigh + WImg
                                neighzmod = neigh_z + WImgz
                                neighymod = neigh_y + WImgy

                            label_OHE, xhat, maxindx = self.model(neighmod, neighzmod, neighymod)

                        val_truncloss = self.criterion(label_OHE, y, indices)

                        # label_losses.append(val_truncloss.detach().data)

                        val_train_loss += val_truncloss.item()
                        _, predicted = torch.max(label_OHE.data, 1)
                        val_total += y.size(0)
                        val_correct += predicted.eq(y.data).cpu().sum()
                        val_correct = val_correct.item()

                        if idx != 0 and idx % 10 == 0:
                            txt = 'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                            val_train_loss / (idx + 1), 100. * val_correct / val_total, val_correct, val_total)
                            print(txt)
                    print("Validation end")
                self.set_mode('train')
        print("[*] Training Finished!")

    def test(self, intimgname, intoutname, affine, dataloader= None, batchsize =1000, modifiedimgs=None,
             imgs = None, uncertfn = None):
        self.set_mode('eval')
        X_list = []
        labeled_list = []
        fn_modImgs = []
        fn_modImgAddeds =[]
        self.xhat_list = []
        if dataloader is not None:
            datanum = 1
        elif imgs is not None:
            datanum = len(imgs)
        else:
            datanum = len(self.obj)

        for i in np.arange(datanum):
            self.xhats = []
            self.maxindxs = []
            label_OHEs = []
            self.centervox = []
            if dataloader is None:
                ind_dataloader = DataLoader(self.data[i], batch_size=batchsize, shuffle=False,
                                             num_workers=6, drop_last=False)
                size = self.data[i].dataset.dataOrigShape[:3]
                origindices = self.data[i].dataset.indices
            elif imgs is not None:
                if not self.uncertainty:
                    data = []
                    for i in np.arange(datanum):
                        tmpdata = MRDataSet2_noupsample.MRDataSet(pkl_file=imgs[i],
                                                                  transform=transforms.Compose([
                                                                      MRDataSet2_noupsample.ToTensor()
                                                                  ]))
                        data.append(tmpdata)

                elif self.uncertainty:
                    data = []
                    for i in np.arange(len(self.obj)):
                        tmpdata = MRDataSet2_mult_dataset.MRDataSet(pkl_file=imgs[i], pkl_file2=uncertfn[i],
                                                                    transform=transforms.Compose([
                                                                        MRDataSet2_mult_dataset.ToTensor()
                                                                    ]))
                        data.append(tmpdata)
                del tmpdata
                ind_dataloader = DataLoader(data[i], batch_size=batchsize, shuffle=False,
                                            num_workers=6, drop_last=False)
                size = data[i].dataset.dataOrigShape[:3]
                origindices = data[i].dataset.indices
            else:
                ind_dataloader = dataloader
                size = dataloader.dataset.dataset.dataOrigShape[:3]
                origindices = dataloader.dataset.dataset.indices
            #todo: change to running median
            width = ((2*self.pad)+1)**2
            modified_img_size = size # + (width*3+20,)
            modified_img = np.zeros(modified_img_size)
            denom_counter = np.zeros(modified_img_size) + 1e-16
            # denom_counter = np.zeros(size).astype(int)
            for idx, (sample, indices, orig_indices, indx, indz, indy) in enumerate(ind_dataloader):

                if self.uncertainty:
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

                    outputs = self.modelWImg(neigh, neigh_z, neigh_y, neigh2, neigh2_z, neigh2_y)

                    if self.ae:
                        (WImg, WImgz, WImgy, WImgU, WImgzU, WImgyU) = outputs
                        neighmod = WImg
                        neighzmod = WImgz
                        neighymod = WImgy

                    else:
                        (WImg, WImgz, WImgy) = outputs
                        neighmod = neigh + WImg
                        neighzmod = neigh_z + WImgz
                        neighymod = neigh_y + WImgy

                    label_OHE, xhat, maxindx = self.model(neighmod, neighzmod, neighymod)

                else:
                    neighbors = sample['neighbors']
                    neighbors_z = sample['neighbors_z']
                    neighbors_y = sample['neighbors_y']

                    neigh = Variable(neighbors.cuda(), requires_grad=False)
                    neigh_z = Variable(neighbors_z.cuda(), requires_grad=False)
                    neigh_y = Variable(neighbors_y.cuda(), requires_grad=False)
                    label_OHE, xhat, maxindx = self.model(neigh, neigh_z, neigh_y)

                    if self.iterative:
                        neighmod, neighzmod, neighymod = self.modelWImg(neigh, neigh_z, neigh_y)
                        label_OHE_mod, xhat_mod, maxindx = self.model(neighmod, neighzmod, neighymod)

                label_OHEs.append(label_OHE.detach())

                # self.Zs.append(Z_dec)
                self.xhats.append(xhat.detach())
                self.neighmod = WImg.detach() # neighmod.detach()
                self.neighzmod = WImgz.detach() # neighzmod.detach()
                self.neighymod = WImgy.detach() # neighymod.detach()

                width = (2*self.pad)+1
                indx = indx.data.cpu().numpy().reshape((-1,3,width,width))
                indy = indy.data.cpu().numpy().reshape((-1,3,width,width))
                indz = indz.data.cpu().numpy().reshape((-1,3,width,width))
                for iind in range(0,indx.shape[0]):
                    # denom_counter = denom_counter.astype(int)
                    modified_img[tuple(indx[iind])] = np.transpose(self.neighmod.data.cpu().numpy()[iind].reshape((width,width))) # + (denom_counter[tuple(indx[iind])],)
                    modified_img[tuple(indz[iind])] = np.transpose(self.neighzmod.data.cpu().numpy()[iind].reshape((width,width))) # + (denom_counter[tuple(indz[iind])],)
                    modified_img[tuple(indy[iind])] = np.transpose(self.neighymod.data.cpu().numpy()[iind].reshape((width,width))) # + (denom_counter[tuple(indy[iind])],)
                    denom_counter[tuple(indx[iind])] += 1
                    denom_counter[tuple(indz[iind])] += 1
                    denom_counter[tuple(indy[iind])] += 1

                # self.Z_fs.append(Z_f.detach())
                self.maxindxs.append(maxindx.detach().data)
                self.centervox.append(neighbors[:,0,self.pad, self.pad])

            # modified_img /= ((2*self.pad)+1)**2
            modified_img /= denom_counter
            # modified_img = np.median(modified_img, axis=3)
            labeled = torch.cat(label_OHEs, 0)
            labeled = labeled.data.cpu().numpy()
            maxindices = torch.cat(self.maxindxs, 0).cpu().numpy()
            centervoxes = torch.cat(self.centervox, 0).cpu().numpy()

            X = np.zeros(size)
            # X_mod = np.zeros(size)
            b = np.asarray(list(itertools.chain.from_iterable(self.xhats)))
            values = np.zeros([b.shape[0], self.f_dim])
            for B in np.arange(len(b)):
                values[B] = b[B].data.cpu().numpy()

            size4d = size + (self.f_dim,)
            Y = np.zeros(size4d)
            size3d = size + (self.labels,)
            L = np.zeros(size3d)
            C = np.zeros(size)

            for idx in np.arange(maxindices.shape[0]):
                idxs = np.unravel_index(origindices[idx], size)
                X[idxs] = maxindices[idx] + 1
                Y[idxs] = values[idx]
                L[idxs] = labeled[idx]
                # X_mod[idxs] = maxindices_mod[idx] + 1
                C[idxs] = centervoxes[idx]

            recon = nib.Nifti1Image(Y, affine=affine[i])
            nib.save(recon, filename=intimgname[i])

            recon = nib.Nifti1Image(X, affine=affine[i])
            nib.save(recon, filename=intoutname[i])

            # recon = nib.Nifti1Image(X_mod, affine=affine[i])
            # nib.save(recon, filename='/data/infant/outputs/modified_img_maxidx_{0}.nii.gz'.format(
            #     datetime.today().strftime('%Y%m%d%h%m%s')))

            # modmin = np.abs(np.min(modified_img))
            # modified_img[ np.where(X>0)]+=modmin
            recon = nib.Nifti1Image(modified_img, affine[i])
            if self.uncertainty:
                suffix = 'freeze_model'
            else:
                suffix = 'init'
            fn_modImg ='/data/infant/modified_imgs/modified_img_{0}.nii.gz'.format(
                datetime.today().strftime('%Y%m%d%h%m%s'))
            nib.save(recon, filename=fn_modImg)

            recon = nib.Nifti1Image(modified_img+C, affine[i])
            fn_modImgAdded = '/data/infant/modified_imgs/modified_img_added_{0}.nii.gz'.format(
                datetime.today().strftime('%Y%m%d%h%m%s'))
            nib.save(recon, filename=fn_modImgAdded)

            fn_modImgs.append(fn_modImg)
            fn_modImgAddeds.append(fn_modImgAdded)

            # labeled_list.append(L.reshape(np.prod(size3d[:3]), 3))
            labeled_list.append(label_OHE)

        #     X_list.append(X)
        #     self.xhat_list.append(self.xhats)
        #
        #
        # self.set_mode('train')

        # return X_list
        return (labeled_list, fn_modImgs, fn_modImgAddeds)



