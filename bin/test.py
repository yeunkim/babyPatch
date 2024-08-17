import run_two_stage_cnn
import pickle
import nibabel as nib
import MRDataSet2_noupsample
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import itertools
import numpy as np
import data_preproc_noupsample
import MRDataSet2_mult_dataset
import uncertainty.uncertainty as uncertainty
import run_two_stage_cnn_h5
import torch
import data_preproc_h5_light
d=1

starttime = time.time()

uncertaintytest = True

numchannels = 5
###### models
# initmodel = '/data/infant/vae_objs/002_1d_T2w_bs_augaug_ants_norm_k512n_e7_lr5e4_f{0}_diceyz.obj'.format(numchannels)
# initmodel = '/data/infant/vae_objs/002_1d_T2wT1w_bs_augaug_ants_norm_k512n_e5_lr1e3_3c_p3_yzxline5_nzy5_7.obj'
# initmodel = '/data/infant/vae_objs/002_1d_T2w_bs_augaug_ants_norm_k512n_e5_lr5e4_f5_norot.obj'
initmodel = '/data/infant/vae_objs/002_1d_T2w_bs_augaug_ants_norm_k512n_e5_lr5e4_f{0}_en3.obj'.format(numchannels)

# refmodel = '/data/infant/vae_objs/002_1d_T2w_bs_ants_norm_k512n_e5_lr5e4_f{0}_{0}ch_nodice.obj'.format(numchannels)
# refmodel = '/data/infant/vae_objs/002_1d_T2wT1w_bs_augaug_ants_norm_k512n_e5_lr1e3_3c_p3_yzxline5_nzy5_7_3channels.obj'
# refmodel = '/data/infant/vae_objs/002_1d_T2w_bs_ants_norm_k512n_e7_lr5e4_f5_5ch_norot.obj'
refmodel = '/data/infant/vae_objs/002_1d_T2w_bs_ants_norm_k512n_e10_lr5e4_f{0}_{0}ch_en3_uncertain.obj'.format(numchannels)

# subjs = ['025', '031', '036', '027', '039', '040']
# subjs = ['040']
subj = '025'
# for subj in subjs:

#### Load obj = Single test data subject
h5file = '/data/infant/h5data/train_raw/{0}_1mm_light.h5'.format(subj)
obj1ch = '/data/infant/vae_objs/{0}_T2w_light_3c_unraveledidx_p3_yzxline5_nzy5_norm.obj'.format(subj)
# obj1ch = '/data/infant/vae_objs/{0}_T2w_1mm.obj'.format(subj)
# obj1ch = '/data/infant/processed/train_data/{0}/{0}_T2w_train_morecsf.obj'.format(subj)
#### image affine matrix
nii = nib.load('/data/T1{0}-5/{0}-skullstripped_anat.nii'.format(subj))._affine
nii = nib.load('/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.N4.cerebrum.nii.gz'.format(subj))._affine
# nii = nib.load('/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.bse.N4.nii.gz'.format(subj))._affine
# nii = nib.load('/ifs/tmp/mmatern/{0}_train_re/{0}_re_T2w.bse.N4.cerebrum.nii.gz'.format(subj))._affine
#### init output names
# initinterfeatimg = '/data/infant/processed/test/{0}_{1}channel_test_x_noz.dice.nii.gz'.format(subj, numchannels)
# initoutput = '/data/infant/processed/test/{0}_{1}channel_test_predicted_noz.dice.nii.gz'.format(subj, numchannels)
initinterfeatimg = '/data/infant/processed/test_data/{0}/{0}_{1}ch_initinterfeatimg.nii.gz'.format(subj, numchannels)
# initinterfeatimg = '/data/infant/processed/test/{0}_5channel_test_x_noz_5ch.nii.gz'.format(subj)
initoutput = '/data/infant/processed/test_data/{0}/{0}_{1}ch_initoutput.nii.gz'.format(subj, numchannels)
#### cerebrum mask
cerebrum_mask = '/data/T1{0}-5/{0}-skullstripped_anat.nii'.format(subj)
cerebrum_mask = '/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj)
# cerebrum_mask = '/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.mask.nii.gz'.format(subj)
# cerebrum_mask = '/ifs/tmp/mmatern/{0}_train_re/{0}_re_T2w.bse.N4.cerebrum.mask.nii.gz'.format(subj)
#### 3-channel interm obj
# obj3ch = '/data/infant/vae_objs/{0}_T2w_ants_norm_{1}channel_noz.dice.obj'.format(subj, numchannels)
# obj3ch = '/data/infant/processed/train_data/{0}/{0}_T2w_train_morecsf_obj3ch.obj'.format(subj)
obj3ch = '/data/infant/vae_objs/{0}_T2w_val_obj{1}ch.obj'.format(subj, numchannels)
#### refine output names
refineinterfeatimg = '/data/infant/processed/test_data/{0}/{0}_refineinterfeatimg_{1}ch.nii.gz'.format(subj, numchannels)
refineoutput = '/data/infant/processed/test_data/{0}/{0}_refineoutput_{1}ch.nii.gz'.format(subj, numchannels)

## uncertainty file
uncertfn = '/data/infant/vae_objs/{0}_vars_10.obj'.format(subj)

## fns
fn4 = '/data/infant/h5data/train_raw/002_erode_edit10_1mm_light.h5'
fn5 = '/data/infant/h5data/train_raw/002_nobias_edit10_1mm_light.h5'
fn6 = '/data/infant/h5data/train_raw/002_bias_edit10_1mm_light.h5'
fns = [fn4, fn5, fn6]
#### param settings

multiinput=False
threedim=False

################################################################################################
############ START ################################################

# file_obj = open(obj1ch, 'rb')
# test1 = pickle.load(file_obj)

if uncertaintytest:
    import MRDataSet_h5

    epoch = 5
    solver = run_two_stage_cnn_h5.Solver([fns[0]], epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=1000,
                                         in_features=1, labels=3, shuffle=True,
                                         channels=1, coords=False, DL=False, width=3, softdiceloss=False, dropout=False)
    solver.model.load_state_dict(torch.load(
        '/data/infant/checkpoints/init_e{1}_lr5e4_f{0}_checkpoint.pth'.format(numchannels, epoch)))

    # testdata = MRDataSet_h5.MRDataSet('/data/infant/test3.hdf5',
    #                                   transform=transforms.Compose([
    #                                       MRDataSet_h5.ToTensor(multiinput=multiinput,
    #                                                                      threedim=threedim)
    #                                   ]), multiinput=multiinput, threedim=threedim)
    #
    # # testdata = MRDataSet2_noupsample.MRDataSet(pkl_file=obj1ch,
    # #                                            transform=transforms.Compose([
    # #                                                MRDataSet2_noupsample.ToTensor(multiinput=multiinput,
    # #                                                                               threedim=threedim)
    # #                                            ]), multiinput=multiinput, threedim=threedim)
    #
    #
    # dataloader = DataLoader(testdata, batch_size=1000, shuffle=False,
    #                         num_workers=0, drop_last=False)
    # del testdata
    iterations = 1
    mean = []
    var = []
    import misc_test
    for i in np.arange(iterations):
        label_OHE = solver.test([initinterfeatimg], [initoutput], [nii], h5file=h5file, pklfile='/data/infant/h5data/train_raw/002_erode_edit10_1mm_light.obj')
        # label_OHE = misc_test.uncertest(solver.model, initinterfeatimg, initoutput, nii, dataloader=dataloader, uncertainty=False)
        # mean, var = uncertainty.compute_var_mean(label_OHE, mean, var, i)
        # del solver

    # size = testdata.dataset.dataOrigShape
    # uncertainty.render_uncert_imgs([obj1ch], var, [nii], [subj], numchannels, iterations)


    ##### perform image pre-processing on the intermediate feature image
    obj = '/data/infant/h5data/{2}_{0}ch_initinterfeatimg_e{1}.h5'.format(numchannels, epoch, subj)
    data_preproc_h5_light.imagepatches(
        fname=initinterfeatimg,
        mask=cerebrum_mask,
        label=cerebrum_mask, fnoutput= obj,
        gm=2, wm=1, csf=3, num_classes=4, channels=numchannels,
        pad=5, normalize=False
    )

    uncert = '/data/infant/h5data/{2}_{0}ch_initoutput_e{1}.h5'.format(numchannels, epoch, subj)
    data_preproc_h5_light.imagepatches(
        fname=initoutput,
        mask=cerebrum_mask,
        label=cerebrum_mask, fnoutput= uncert,
        gm=2, wm=1, csf=3, num_classes=4, channels=3,
        pad=5, normalize=False
    )

    elapsed = time.time() - starttime

    ###### load intermediate feature image
    # file_obj = open(obj3ch, 'rb')
    # test1 = pickle.load(file_obj)

    ############# Load refinement model
    # model_obj = open(refmodel, 'rb')
    # solver = pickle.load(model_obj)
    epoch2 = 7
    solver = run_two_stage_cnn_h5.Solver([obj], epoch=epoch2, lr=5e-4, f_dim=numchannels, batch_size=1000,
                                         in_features=1,
                                         labels=3,
                                         shuffle=True, channels=numchannels, coords=False, DL=False, width=3,
                                         softdiceloss=False,
                                         uncertainty=True, uncertfn=[uncert], channels2=1)
    solver.model.load_state_dict(torch.load('/data/infant/checkpoints/ref_e{0}_lr5e4_f{1}_checkpoint.pth'.format(epoch2, numchannels)))
    label_OHE = solver.test([refineinterfeatimg], [refineoutput], [nii], h5file=obj, h5file2=uncert)
    # label_OHE = misc_test.uncertest(solver.model, refineinterfeatimg, refineoutput, nii, dataloader=dataloader, uncertainty=True)
    # size = test1.dataOrigShape[:3]
    # X = solver.test(dataloader, size, test1.indices, test1.dataUpsampledShape, test1.patchsize)

