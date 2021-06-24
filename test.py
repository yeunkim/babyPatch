import run_two_stage_cnn_orig
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
import uncertainty
import run_two_stage_cnn_orig
import torch
import data_preproc_h5_light
d=1

starttime = time.time()

uncertaintytest = True

numchannels = 4

# subjs = ['039', '023', '132','002','087', '115','056','072','010','108']
subjs = ['010','115','056','132','023']

Ttype= 'T2_N4'
suffix = 'T2_N4'
for subj in subjs:
    #### Load obj = Single test data subject
    # h5file = '/mnt/data/infant/h5data/train_raw/{0}_ibeatspace_1mm.obj'.format(subj)
    # h5file = '/mnt/data/infant/h5data/train_raw/025_{0}_1mm_test.obj'.format(Ttype)
    h5file = '/data/infant/objects/{0}_N4_1mm.obj'.format(subj)
    # h5file = '/mnt/data/infant/h5data/train_raw/{0}_1mm_ibeat.obj'.format(subj)
    # obj1ch = '/mnt/data/infant/vae_objs/{0}_T2w_light_3c_unraveledidx_p3_yzxline5_nzy5_norm.obj'.format(subj)
    # obj1ch = '/mnt/data/infant/vae_objs/{0}_T2w_1mm.obj'.format(subj)
    # obj1ch = '/mnt/data/infant/processed/train_data/{0}/{0}_T2w_train_morecsf.obj'.format(subj)
    #### image affine matrix
    # nii = nib.load('/mnt/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.N4.cerebrum.nii.gz'.format(subj))._affine
    # nii = nib.load('/mnt/data/infant/processed/test_data/{0}/{0}_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj))._affine
    # nii = nib.load('/data/infant/cerebrum_T2/{0}-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj))._affine
    # nii = nib.load('//nafs/shattuck/yeunkim/infant_images/rebeccabelisle/{0}-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj))._affine
    nii = nib.load('//data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subj))._affine
    # nii = nib.load('/mnt/data/infant/processed/train_data/{0}/{0}_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj))._affine
    # nii = nib.load('/mnt/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.bse.N4.nii.gz'.format(subj))._affine
    # nii = nib.load('/ifs/tmp/mmatern/{0}_train_re/{0}_re_T2w.bse.N4.cerebrum.nii.gz'.format(subj))._affine
    # nii = nib.load('/kahlo/data/T1{0}-5/{0}-skullstripped_anat.nii'.format(subj))._affine
    #### init output names
    initinterfeatimg = '/data/infant/intermediate_nii/{0}_{1}ch_initinterfeatimg_{2}.nii.gz'.format(subj, numchannels,suffix)
    # initoutput = '/mnt/data/infant/processed/test/{0}_{1}channel_test_predicted_noz.dice.nii.gz'.format(subj, numchannels)
    # initinterfeatimg = '/data/infant/processed/test_data/{0}/{0}_{1}ch_initinterfeatimg.nii.gz'.format(subj, numchannels)
   # initinterfeatimg = '/data/infant/processed/test_data/TD2/{0}/{0}_{1}ch_initinterfeatimg0.nii.gz'.format(subj,
                                                                                                      # numchannels)
    # initinterfeatimg = '/mnt/data/infant/processed/train_data/{0}/{0}_{1}ch_initinterfeatimg.nii.gz'.format(subj, numchannels)
    # initinterfeatimg = '/mnt/data/infant/processed/test/{0}_5channel_test_x_noz_5ch.nii.gz'.format(subj)
    initoutput = '/data/infant/intermediate_nii/{0}_{1}ch_initoutput_{2}.nii.gz'.format(subj, numchannels,suffix)
    # initoutput = '/data/infant/processed/test_data/TD2/{0}/{0}_{1}ch_initoutput0.nii.gz'.format(subj, numchannels)
    # initoutput = '/mnt/data/infant/processed/train_data/{0}/{0}_{1}ch_initoutput.nii.gz'.format(subj, numchannels)
    #### cerebrum mask
    # cerebrum_mask = '/mnt/data/T1{0}-5/{0}-skullstripped_anat.nii'.format(subj)
    # cerebrum_mask = '/data/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj)
    # cerebrum_mask = '/data/infant/cerebrum_T2/{0}-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj)
    # cerebrum_mask = '//nafs/shattuck/yeunkim/infant_images/rebeccabelisle/{0}-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj)
    cerebrum_mask = '//data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subj)
    # cerebrum_mask = '/data/infant/processed/test_data/{0}/{0}_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj)
    # cerebrum_mask = '/kahlo/data/T1{0}-5/{0}-skullstripped_anat.nii'.format(subj)
    # cerebrum_mask = '/mnt/data/infant/processed/train_data/{0}/{0}_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj)
    # cerebrum_mask = '/mnt/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.mask.nii.gz'.format(subj)
    # cerebrum_mask = '/ifs/tmp/mmatern/{0}_train_re/{0}_re_T2w.bse.N4.cerebrum.mask.nii.gz'.format(subj)
    # cerebrum_mask = '/mnt/data/{0}-skullstripped_anat.mask.nii'.format(subj)
    #### 3-channel interm obj
    # obj3ch = '/mnt/data/infant/vae_objs/{0}_T2w_ants_norm_{1}channel_noz.dice.obj'.format(subj, numchannels)
    # obj3ch = '/mnt/data/infant/processed/train_data/{0}/{0}_T2w_train_morecsf_obj3ch.obj'.format(subj)
    # obj3ch = '/mnt/data/infant/vae_objs/{0}_T2w_val_obj{1}ch.obj'.format(subj, numchannels)
    #### refine output names
    refineinterfeatimg = '/data/infant/outputs/{0}_refineinterfeatimg_{1}ch_{2}.nii.gz'.format(subj, numchannels,suffix)
    refineoutput = '/data/infant/outputs/{0}_refineoutput_{1}ch_{2}.nii.gz'.format(subj, numchannels,suffix)
    # refineinterfeatimg = '/mnt/data/infant/processed/train_data/{0}/{0}_refineinterfeatimg_{1}ch.nii.gz'.format(subj, numchannels)
    # refineoutput = '/mnt/data/infant/processed/train_data/{0}/{0}_refineoutput_{1}ch.nii.gz'.format(subj, numchannels)


    ## uncertainty file
    uncertfn = '/data/infant/vae_objs/{0}_vars_10.obj'.format(subj)

    ## fns
    # ext = 'obj'
    # light = ''
    # fn4 = '/mnt/data/infant/h5data/train_raw/002_erode_edit10_1mm_light.h5'
    # fn5 = '/mnt/data/infant/h5data/train_raw/002_nobias_edit10_1mm_light.h5'
    # fn6 = '/mnt/data/infant/h5data/train_raw/002_bias_edit10_1mm_light.h5'
    # fn4 = '/mnt/data/infant/h5data/train_raw/002_erode_edit10_1mm{1}.{0}'.format(ext, light)
    # fn5 = '/mnt/data/infant/h5data/train_raw/002_nobias_edit10_1mm{1}.{0}'.format(ext, light)
    # fn6 = '/mnt/data/infant/h5data/train_raw/002_bias_edit10_1mm{1}.{0}'.format(ext, light)
    # fns = [fn4, fn5, fn6]
    fns = [h5file]
    #### param settings

    multiinput=False
    threedim=False

################################################################################################
############ START ################################################

# file_obj = open(obj1ch, 'rb')
# test1 = pickle.load(file_obj)
    if uncertaintytest:
        import MRDataSet_h5
        iterative= True
        mean = []
        var = []
        iterations = 2
        epoch = 5
        for i in np.arange(iterations):
            solver = run_two_stage_cnn_orig.Solver([fns[0]], epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=1000,
                                                 in_features=1, labels=3, shuffle=True,
                                                 channels=1, coords=False, DL=False, softdiceloss=False, dropout=False)
            #solver.model.load_state_dict(torch.load(
            #    '/mnt/data/infant/checkpoints/init_e{1}_lr5e4_f{0}_i{2}_checkpoint_probmean.pth'.format(numchannels, epoch, i)))
            solver.model.load_state_dict(torch.load('/oldmiro/data/SSD_data/infant/checkpoints/init_e10_lr5e4_f4_i3_checkpoint_3input_6data.pth'))

            label_OHE = solver.test([initinterfeatimg], [initoutput], [nii], batchsize=10000)
            # label_OHE = misc_test.uncertest(solver.model, initinterfeatimg, initoutput, nii, dataloader=dataloader, uncertainty=False)
            mean, var = uncertainty.compute_var_mean(label_OHE, mean, var, i)
                # del solver

        # size = testdata.dataset.dataOrigShape
        uncertainty.render_uncert_imgs([h5file], var, [nii], [subj], numchannels, iterations, mean=mean, pkl=True, suffix=suffix)


        ##### perform image pre-processing on the intermediate feature image
        obj = '/data/infant/objects/{2}_{0}ch_initinterfeatimg_e{1}_{3}'.format(numchannels, epoch, subj,suffix)
        # data_preproc_h5_light.imagepatches(
        #     fname=initinterfeatimg,
        #     mask=cerebrum_mask,
        #     label=cerebrum_mask, fnoutput= obj,
        #     gm=2, wm=1, csf=3, num_classes=4, channels=numchannels,
        #     pad=5, normalize=False
        # )
        data0 = data_preproc_noupsample.imagepatches(
            fname=initinterfeatimg,
            mask=cerebrum_mask,
            label=cerebrum_mask,
            gm=2, wm=1, csf=3, num_classes=4, channels=numchannels,
            masklabel=True, pad=5, normalize=False)
        file_obj = open('{0}.obj'.format(obj), 'wb')
        pickle.dump(data0, file_obj, protocol=4)
        file_obj.close()

        uncertnii = '/data/infant/variance/{0}_vars_i{2}_{1}ch_en_0_{3}.nii.gz'.format(subj, numchannels, iterations, suffix)
        uncert = '/data/infant/objects/{2}_{0}ch_uncert_e{1}_i{3}_{4}'.format(numchannels, epoch, subj, iterations,suffix)
        # data_preproc_h5_light.imagepatches(
        #     fname=uncertnii,
        #     mask=cerebrum_mask,
        #     label=cerebrum_mask, fnoutput= uncert,
        #     gm=2, wm=1, csf=3, num_classes=4, channels=3,
        #     pad=5, normalize=False
        # )
        data0 = data_preproc_noupsample.imagepatches(
            fname=uncertnii,
            mask=cerebrum_mask,
            label=cerebrum_mask,
            gm=2, wm=1, csf=3, num_classes=4, channels=3,
            pad=5, normalize=False)
        file_obj = open('{0}.obj'.format(uncert), 'wb')
        pickle.dump(data0, file_obj, protocol=4)
        file_obj.close()

        # probmeannii ='/data/infant/{0}_means_i{2}_{1}ch_en_0_{3}.nii.gz'.format(subj, numchannels, iterations, suffix)
        # probmean = '/data/infant/{3}_{0}ch_probmean_e{1}_i{2}_{4}'.format(numchannels, epoch,  iterations,
        #                                                                               subj, suffix)
        # data0 = data_preproc_noupsample.imagepatches(
        #     fname=initoutput,
        #     mask=cerebrum_mask,
        #     label=cerebrum_mask,
        #     gm=2, wm=1, csf=3, num_classes=4, # channels=3,
        #     masklabel=True, pad=5, normalize=False)
        # file_obj = open('{0}.{1}'.format(probmean, 'obj'), 'wb')
        # pickle.dump(data0, file_obj, protocol=4)



        ###### load intermediate feature image
        # file_obj = open(obj3ch, 'rb')
        # test1 = pickle.load(file_obj)

        ############# Load refinement model
        # model_obj = open(refmodel, 'rb')
        # solver = pickle.load(model_obj)
        epoch2 = 5
        solver = run_two_stage_cnn_orig.Solver(['{0}.obj'.format(obj)], epoch=epoch2, lr=5e-4, f_dim=numchannels, batch_size=1000,
                                             in_features=1,
                                             labels=3,
                                             shuffle=True, channels=numchannels, coords=False, DL=False,
                                             softdiceloss=False,
                                             uncertainty=True, uncertfn=['{0}.obj'.format(uncert)], channels2=3
                                             )
        # solver.model.load_state_dict(torch.load(
        #     '/mnt/data/infant/checkpoints/ref_e{0}_lr5e4_f{1}_i{2}_checkpoint_probmean.pth'.format(epoch2, numchannels,iterations)))
        solver.model.load_state_dict(torch.load(
            '/oldmiro/data/SSD_data/infant/checkpoints/ref_e10_lr5e4_f4_i4_checkpoint_3input_6data.pth'))

        for i in np.arange(iterations):
            label_OHE = solver.test([refineinterfeatimg], [refineoutput], [nii],
                                    batchsize=10000)
            mean, var = uncertainty.compute_var_mean(label_OHE, mean, var, i)
        uncertainty.render_uncert_imgs([h5file], var, [nii], [subj+'_ref'], numchannels, iterations, mean=mean, pkl=True)
        # label_OHE = misc_test.uncertest(solver.model, refineinterfeatimg, refineoutput, nii, dataloader=dataloader, uncertainty=True)
        # size = test1.dataOrigShape[:3]
        # X = solver.test(dataloader, size, test1.indices, test1.dataUpsampledShape, test1.patchsize)

elapsed = time.time() - starttime
print(elapsed)