import run_two_stage_cnn
import pickle
import nibabel as nib
import MRDataSet2_noupsample
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import itertools
import numpy as np
import MRDataSet2_mult_dataset
import data_preproc_noupsample
import uncertainty
import run_two_stage_cnn_h5
import torch
import data_preproc_h5
import data_preproc_h5_light
d=1

numchannels = 5
# #### fn = All training data object
fn4 = '/data/infant/h5data/train_raw/002_erode_edit10_1mm_light.h5'
fn5 = '/data/infant/h5data/train_raw/002_nobias_edit10_1mm_light.h5'
fn6 = '/data/infant/h5data/train_raw/002_bias_edit10_1mm_light.h5'
subj = '002'
# label = '/data/infant/processed/train_data/002-T2.MRM.YK.label.nii.gz'
nii = nib.load('/data/infant/processed/train_data/002/002-T2w_resampled.N4.cerebrum.nii.gz')._affine
# #### param settings
multiinput=False
threedim=False
softdiceloss = False

###############################################################################################
########### START ################################################

starttime = time.time()

dropout= False
iterations = 1

mean = []
var = []

fns = [fn4, fn5, fn6]
# dir='/data/infant/h5data/train_raw'
epoch = 1
initinterfeatimgs = []
initoutputs = []
epoch2 = 10
for i in np.arange(len(fns)):
    # solver = run_two_stage_cnn_h5.Solver([fns[i]], epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=10000,
    #                                      in_features=1, labels=3, shuffle=True,
    #                                   channels=1,coords=False, DL=False, width=3,
    #                                      softdiceloss=softdiceloss, dropout=dropout,
    #                                      valobj='/data/infant/vae_objs/010_val.obj'
    #                                      )
    # solver.train()
    initinterfeatimg = '/data/infant/processed/test/002_{0}ch_initinterfeatimg_e{1}_fn{2}.nii.gz'.format(numchannels, epoch, i)
    initoutput = '/data/infant/processed/test/002_{0}ch_initoutput_e{1}_f{2}.nii.gz'.format(numchannels, epoch, i)
    initinterfeatimgs.append(initinterfeatimg)
    initoutputs.append(initoutput)

    # label_OHE = solver.test([initinterfeatimg], [initoutput], [nii], batchsize=10000)
    # mean, var = uncertainty.compute_var_mean(label_OHE, mean, var, i)
    # del solver.data, solver.dataloader

solver = run_two_stage_cnn_h5.Solver(fns, epoch=epoch2, lr=5e-4, f_dim=numchannels, batch_size=1000,
                                     in_features=1, labels=3, shuffle=True,
                                     channels=1, coords=False, DL=False, width=3, softdiceloss=softdiceloss,
                                     dropout=dropout,valobj='/data/infant/vae_objs/010_val.obj')
solver.train()
for i in np.arange(len(fns)):
    initinterfeatimg = '/data/infant/processed/test/002_{0}ch_initinterfeatimg_e{1}_fn{2}.nii.gz'.format(numchannels, epoch2, i)
    initoutput = '/data/infant/processed/test/002_{0}ch_initoutput_e{1}_f{2}.nii.gz'.format(numchannels, epoch2, i)
    initinterfeatimgs.append(initinterfeatimg)
    initoutputs.append(initoutput)
label_OHE = solver.test(initinterfeatimgs[3:], initoutputs[3:], [nii,nii,nii], batchsize=10000)
modelpath = '/data/infant/checkpoints/init_e{1}_lr5e4_f{0}.pth'.format(numchannels, epoch2)
del solver.data, solver.dataloader
torch.save(solver, modelpath)
## Save general checkpoint
torch.save(solver.model.state_dict(), '/data/infant/checkpoints/init_e{1}_lr5e4_f{0}_checkpoint.pth'.format(numchannels, epoch2))

# uncertainty.render_uncert_imgs(fns, var, [nii, nii, nii, nii], [subj, subj, subj, subj], numchannels, iterations)
elapsed = time.time() - starttime
#
del solver

##### perform image pre-processing on the intermediate feature image
cerebrum_mask0 =  '/data/infant/processed/train_data/002/002-T2w_resampled.cerebrum.mask.nii.gz'
cerebrum_mask1 = '/data/infant/processed/train_data/002/002-T2w_resampled.cerebrum.mask.erode.nii.gz'
label = '/data/infant_2019/transformed_labels/002/002.uncor.label.nii.gz'
masks = [cerebrum_mask1 ,cerebrum_mask0, cerebrum_mask0, cerebrum_mask1 ,cerebrum_mask0, cerebrum_mask0]

# uncertnii = '/data/infant/processed/test/{0}_vars_i{2}_{1}ch_en_{3}.nii.gz'.format(subj, numchannels, iterations, 1)
# uncertnii2 = '/data/infant/processed/test/{0}_vars_i{2}_{1}ch_en_{3}.nii.gz'.format(subj, numchannels, iterations, 2)
# uncertnii3 = '/data/infant/processed/test/{0}_vars_i{2}_{1}ch_en_{3}.nii.gz'.format(subj, numchannels, iterations, 3)
# uncertnii4 = '/data/infant/processed/test/{0}_vars_i{2}_{1}ch_en_{3}.nii.gz'.format(subj, numchannels, iterations, 4)
# uncertniis = [uncertnii, uncertnii2, uncertnii3,uncertnii4]
objs = []
uncerts = []

for i in np.arange(0,len(initinterfeatimgs)):
    obj = '/data/infant/h5data/002_{0}ch_initinterfeatimg_e{1}_fn{2}'.format(numchannels, epoch, i)
    data_preproc_h5_light.imagepatches(
        fname=initinterfeatimgs[i],
        mask=masks[i],
        label=label, fnoutput= obj,
        gm=2, wm=1, csf=3, num_classes=4, channels=numchannels,
        masklabel=True, pad=5, normalize=False
    )
    objs.append('{0}.h5'.format(obj))

    uncert = '/data/infant/h5data/002_{0}ch_initoutput_e{1}_f{2}'.format(numchannels, epoch, i)
    data_preproc_h5_light.imagepatches(
        fname=initoutputs[i],
        mask=masks[i],
        label=label, fnoutput= uncert,
        gm=2, wm=1, csf=3, num_classes=4, channels=3,
        masklabel=True, pad=5, normalize=False
    )
    uncerts.append('{0}.h5'.format(uncert))

cont = '/data/infant/vae_objs/002_continuitytest.obj'
epoch2 = 10
solver = run_two_stage_cnn_h5.Solver(objs, epoch=epoch2, lr=5e-4, f_dim=numchannels, batch_size=50000, in_features=1, labels=3,
                                  shuffle=True, channels=numchannels,coords=False, DL=False, width=3, softdiceloss=False,
                                  uncertainty=True, uncertfn= uncerts, channels2=1,valobj='/data/infant/vae_objs/010_val.obj')

solver.train()

refineinterfeatimgs =[]
refineoutputs = []
for i in np.arange(len(objs)):
    refineinterfeatimg = '/data/infant/processed/test/002_{0}ch_refineinterfeatimg_e{1}_fn{2}.nii.gz'.format(numchannels,
                                                                                                         epoch, i)
    refineoutput = '/data/infant/processed/test/002_{0}ch_refineoutput_e{1}_f{2}.nii.gz'.format(numchannels, epoch, i)
    refineinterfeatimgs.append(refineinterfeatimg)
    refineoutputs.append(refineoutput)

label_OHE = solver.test(refineinterfeatimgs, refineoutputs, [nii, nii, nii,nii, nii, nii], batchsize=10000)
print('002 second model prediction finished.')

del solver.data, solver.dataloader
torch.save(solver.model.state_dict(), '/data/infant/checkpoints/ref_e{0}_lr5e4_f{1}_checkpoint.pth'.format(epoch2, numchannels))

