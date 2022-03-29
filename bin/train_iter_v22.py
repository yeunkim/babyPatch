# skull-stripping


import nibabel as nib
import time
import numpy as np
import uncertainty
import torch
from datetime import datetime
from run import run_two_stage_cnn_orig_truncatedloss, run_classify_weightedImg, run_two_stage_cnn_orig_truncatedloss_v22
from preprocess import generate_wmsubgm_mask, generate_obj_files

## hyperparameters
numchannels = 4
iterations = 2
pad = 5
validation=True
iterative = 1
generator = False
epoch = 3
gm2wm = 0.29 # ratio of number of voxels (gm to wm) [if skull stripping -> put brain:non-brain]
csf2wm = 0 # ratio of number of voxels (csf to wm) [0 if skull stripping]
third_model = False
numslices =(10,10,10)
slices = True
axes = (True,True,True)
dataset_portion = 0.50
pkl = False

## set file names and folder paths
subjs = {'val':['injured24h'], #
        'train':['sham', 'injured24h']}

for numslices in [10,25,50]:
    print('Number of slices: {0}'.format(numslices))
# numslices = 10
    losses_folder = '/data/mouse/losses/'
    checkpoints_folder = '/data/mouse/checkpoints/'
    intermediate_folder = '/data/mouse/intermediate_nii/'
    outputs_folder =  '/data/mouse/outputs/'
    objects_folder = '/data/mouse/objects/'
    fns_whole = ['/{1}/{0}_whole.h5'.format(subjs['train'][i],objects_folder) for i in range(len(subjs['train']))]
    valfns_whole = ['/{1}/{0}_whole.h5'.format(subjs['val'][i],objects_folder) for i in range(len(subjs['val']))]
    # masks = ['/data/mouse/Neil_TBI/T2highres_{0}.mask.nii.gz'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
    niis = [nib.load('/data/mouse/Neil_TBI/T2highres_{0}.mask.nii.gz'.format(subjs['train'][i]))._affine for i in range(len(subjs['train']))]
    # valmasks = ['/data/mouse/Neil_TBI/T2highres_{0}.mask.nii.gz'.format(subjs['val'][i]) for i in range(len(subjs['val']))]
    valniis = [nib.load('/data/mouse/Neil_TBI/T2highres_{0}.mask.nii.gz'.format(subjs['val'][i]))._affine for i in range(len(subjs['val']))]
    labels = ['/data/mouse/Neil_TBI/T2highres_{0}.mask.nii.gz'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
    vallabels = ['/data/mouse/Neil_TBI/T2highres_{0}.mask.nii.gz'.format(subjs['val'][i]) for i in range(len(subjs['val']))]

    mean = []
    var = []
    valmean = []
    valvar = []

    starttime = time.time()
    for ITER in np.arange(iterative):
        print('Starting iteration number {0}'.format(ITER+1))
        # TODO: CHANGE SUFFIX ACCORDING TO RUN
        suffix = '2022_{0}slices'.format(numslices) + str(ITER+1)

        ############################################################################################
        ############################################################################################
        ### train the 1st model
        ############################################################################################
        ############################################################################################
        initinterfeatimgs_alliterations = []
        initoutputs_alliterations = []
        initimodels = []
        for ii in np.arange(iterations):

            textfn = '/{1}/avglosses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),losses_folder)
            with open(textfn, 'w') as f:
                f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
            print('Starting first model training, iteration number {0}, uncertainty iteration {1}'.format(ITER + 1, ii+1))
            solver = run_two_stage_cnn_orig_truncatedloss_v22.Solver(fns_whole, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=22500,
                                            labels=2, shuffle=True, pad=pad, channels=1, textfn = textfn, suffix=suffix, valobj=valfns_whole,
                                            numslices= numslices, slices=slices, axes = axes, dataset_portion=dataset_portion
                                            )
            solver.train()

            # solver.model.load_state_dict(torch.load(
            #     '/{4}/init_e{1}_lr5e4_f{0}_i{2}_{3}.pth'.format(numchannels,int(epoch),0,suffix,checkpoints_folder)))
            # solver.train(epoch=epoch_ext)
            initinterfeatimgs = []
            initoutputs = []
            valinitinterfeatimgs = []
            valinitoutputs = []
            for i in np.arange(len(fns_whole)):
                initinterfeatimg = '/{6}/{4}_{0}ch_initinterfeatimg_e{1}_i{3}_{5}.nii.gz'.format(numchannels, epoch, i, ii, subjs['train'][i], suffix,intermediate_folder)
                initoutput = '/{6}/{4}_{0}ch_initoutput_e{1}_i{3}_f{2}_{5}.nii.gz'.format(numchannels, epoch, i, ii, subjs['train'][i],suffix,intermediate_folder)
                initinterfeatimgs.append(initinterfeatimg)
                initoutputs.append(initoutput)
            if validation:
                for i in np.arange(len(valfns_whole)):
                    valinitinterfeatimg = '/{6}/{4}_{0}ch_valinitinterfeatimg_e{1}_i{3}_{5}.nii.gz'.format(
                        numchannels, epoch, i, ii, subjs['val'][i], suffix,intermediate_folder)
                    valinitoutput = '/{6}/{4}_{0}ch_valinitoutput_e{1}_i{3}_f{2}_{5}.nii.gz'.format(
                        numchannels, epoch, i, ii,subjs['val'][i], suffix,intermediate_folder)
                    valinitinterfeatimgs.append(valinitinterfeatimg)
                    valinitoutputs.append(valinitoutput)
                initinterfeatimgs_alliterations.append(initinterfeatimgs + valinitinterfeatimgs)
                initoutputs_alliterations.append(initoutputs + valinitoutputs)
            else:
                initinterfeatimgs_alliterations.append(initinterfeatimgs)
                initoutputs_alliterations.append(initoutputs)
            initmodel = ('/{4}/init_e{1}_lr5e4_f{0}_i{2}_{3}.pth'.format(numchannels,int(epoch),ii,suffix,checkpoints_folder))
            torch.save(solver.model.state_dict(), initmodel)
            initimodels.append(initmodel)

            label_OHE = solver.test(initinterfeatimgs, initoutputs, niis,batchsize=5000, imgs=fns_whole)

            mean, var = uncertainty.compute_var_mean(label_OHE, mean, var, ii) # running calculation of mean and variance of the estimates
            if validation:
                label_OHE = solver.test(valinitinterfeatimgs, valinitoutputs, valniis, batchsize=5000, imgs=valfns_whole)
                valmean, valvar = uncertainty.compute_var_mean(label_OHE, valmean, valvar, ii)

            del label_OHE


        ############################################################################################
        ############################################################################################
        ### compute uncertainty images
        ############################################################################################
        ############################################################################################
        uncertainty.render_uncert_imgs(fns_whole, var, niis,
                                       subjs['train'], numchannels, iterations,pkl=pkl, suffix=suffix)
        if validation:
            uncertainty.render_uncert_imgs(valfns_whole, valvar, valniis,
                                           subjs['val'], numchannels, iterations, pkl=pkl, suffix='val'+suffix)
        del var, valvar, mean, valmean

        #### Choose which model/intermediate dataset
        initinterfeatimgs_for_refstage, initoutputs_for_refstage, \
        valinitinterfeatimgs_for_refstage, valinitoutputs_for_refstage, smallestidx = uncertainty.select_best_model(gm2wm, csf2wm, iterations,
                                                        initoutputs_alliterations, initimodels, initinterfeatimgs_alliterations,
                                                        valinitinterfeatimgs, validation=True)
        print('First model finished. Generating pickled images, iteration number {0}'.format(ITER + 1))

        ############################################################################################
        ############################################################################################
        ### generate obj files
        ############################################################################################
        ############################################################################################
        uncertniis = []
        valuncertniis = []
        objs = []
        uncerts = []
        valobjs = []
        valuncerts = []
        objs_whole = []
        uncerts_whole = []
        valobjs_whole = []
        valuncerts_whole = []
        for i in np.arange(len(fns_whole)):
            uncertniis.append('/data/infant/variance/{0}_vars_i{2}_{1}ch_en_{3}_{4}.nii.gz'.format(
                subjs['train'][i], numchannels, iterations, i, suffix))
            # fname = generate_obj_files.generate_mask(initinterfeatimgs_for_refstage[i].split('.')[0], fns_whole[i])
            obj = '/{6}/{4}_{0}ch_initinterfeatimg_e{1}_i{3}_{5}'.format(
                numchannels, epoch, i, smallestidx, subjs['train'][i], suffix,objects_folder)
            objs.append('{0}.h5'.format(obj))
            generate_obj_files.generate_h5_files(obj, initinterfeatimgs_for_refstage[i], None, labels[i])
            uncert = '//{6}/{4}_{0}ch_uncert_e{1}_f{2}_i{3}_{5}'.format(numchannels, epoch, i, smallestidx,
                                                                        subjs['train'][i], suffix,objects_folder)
            uncerts.append('{0}.h5'.format(uncert))
            generate_obj_files.generate_h5_files(uncert, uncertniis[i], None, labels[i])

        if validation:
            for i in np.arange(len(valfns_whole)):
                valuncertniis.append(
                    '//data/infant/variance/{0}_vars_i{2}_{1}ch_en_{3}_val{4}.nii.gz'.format(
                        subjs['val'][i], numchannels, iterations, i,suffix))
                # fname = generate_obj_files.generate_mask(valinitinterfeatimgs_for_refstage[i].split('.')[0], valfns_whole[i])
                obj = '//{6}/{4}_{0}ch_valinitinterfeatimg_e{1}_i{3}_{5}'.format(
                    numchannels, epoch,i, smallestidx, subjs['val'][i], suffix,objects_folder)
                valobjs.append('{0}.h5'.format(obj))
                generate_obj_files.generate_h5_files(obj, valinitinterfeatimgs_for_refstage[i],None, vallabels[i])
                uncert = '//{6}/{4}_{0}ch_valuncert_e{1}_f{2}_i{3}_{5}'.format(
                    numchannels, epoch, i,smallestidx,subjs['val'][i], suffix,objects_folder)
                valuncerts.append('{0}.h5'.format(uncert))
                generate_obj_files.generate_h5_files(uncert, valuncertniis[i], None, vallabels[i])


        # ############################################################################################
        # ############################################################################################
        # ## Train the generator
        # ############################################################################################
        # ############################################################################################
        # if generator:
        #     textfn = '/{1}/generator_train_losses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),losses_folder)
        #     with open(textfn, 'w') as f:
        #         f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
        #     solver = run_classify_weightedImg.Solver(fns, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=5000,
        #                                              in_features=1, labels=3, shuffle=True, pad=pad,
        #                                              channels=1, textfn = textfn, uncertainty=True, uncertfn= uncerts,
        #                                              initmodel=initmodel, ae = False, valobj=valfns, valuncertfn=valuncerts)
        #     print('Starting generator training iteration number {0}'.format(ITER + 1))
        #     solver.train()
        #
        #     # solver.model.load_state_dict(torch.load(
        #     #    '/{3}/ref_e{0}_lr5e4_f{1}_i{2}_checkpoint3.pth'.format(epoch2, numchannels, iterations,checkpoints_folder)))
        #
        #     refineinterfeatimgs =[]
        #     refineoutputs = []
        #     for i in np.arange(len(objs)):
        #         refineinterfeatimg = '/{6}/{4}_{0}ch_modifiedinterfeatimg_e{1}_i{3}_{5}.nii.gz'.format(numchannels,
        #                                                                                                              epoch, i, 0, subjs['train'][i],suffix,outputs_folder)
        #         refineoutput = '/{6}/{4}_{0}ch_modifiedoutput_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels, epoch, i, 0,
        #                                                                                                          subjs['train'][i],suffix,outputs_folder)
        #         refineinterfeatimgs.append(refineinterfeatimg)
        #         refineoutputs.append(refineoutput)
        #
        #     (label_OHE, fn_mods, fn_mod_adds) = solver.test(refineinterfeatimgs, refineoutputs, niis, batchsize=1000)
        #
        #     if validation:
        #         valrefineinterfeatimgs = []
        #         valrefineoutputs = []
        #         for i in np.arange(len(valobjs)):
        #             valrefineinterfeatimg = '/{6}/{4}_{0}ch_valmodifiedinterfeatimg_e{1}_i{3}_{5}.nii.gz'.format(
        #                 numchannels,epoch, i, 0, subjs['val'][i], suffix,outputs_folder)
        #             valrefineoutput = '/{6}/{4}_{0}ch_valmodifiedoutput_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels,epoch, i, 0,
        #                                                                                                              subjs['val'][i],suffix,outputs_folder)
        #             valrefineinterfeatimgs.append(valrefineinterfeatimg)
        #             valrefineoutputs.append(valrefineoutput)
        #
        #         (_, _, val_fn_mod_adds) = solver.test(valrefineinterfeatimgs, valrefineoutputs, niis, batchsize=1000)
        #
        #     fns = []
        #     valfns=[]
        #     print('Generating picked images for modified images, iteration number {0}'.format(ITER + 1))
        #     for i,fn_mod_add in enumerate(fn_mod_adds):
        #         prefix = fn_mod_add.split('.')[0]
        #         obj = prefix.split('modified_imgs')[0] + 'objects' + prefix.split('modified_imgs')[1]
        #         generate_obj_files.generate_obj_files(obj, fn_mod_add, masks[i], labels[i])
        #         fns.append(obj+'.obj')
        #     if validation:
        #         for i, val_fn_mod_add in enumerate(val_fn_mod_adds):
        #             prefix = val_fn_mod_add.split('.')[0]
        #             obj = prefix.split('modified_imgs')[0] + 'objects' + prefix.split('modified_imgs')[1]
        #             generate_obj_files.generate_obj_files(obj, val_fn_mod_add, valmasks[i], vallabels[i])
        #             valfns.append(obj+'.obj')
        #
        #     torch.save(solver.modelWImg.state_dict(), '/data/infant/checkpoints/gen_e{0}_lr5e4_f{1}_checkpoint_modelWImg_{2}.pth'.format(epoch, numchannels,suffix))

        ############################################################################################
        ############################################################################################
        ### train the 2nd stage model
        ############################################################################################
        ############################################################################################
        # path = '/data/infant/objects/'
        # objs = [path + '023_4ch_initinterfeatimg_e1_i0_2022_10slices1.h5']
        # uncerts = [path + '023_4ch_uncert_e1_f0_i0_2022_10slices1.h5']
        # valobjs = [path + '010_4ch_valinitinterfeatimg_e1_i0_2022_10slices1.h5']
        # valuncerts = [path + '010_4ch_valuncert_e1_f0_i0_2022_10slices1.h5' ]

        textfn = '/{1}/secondmodel_train_losses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),losses_folder)

        with open(textfn, 'w') as f:
            f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
        solver = run_two_stage_cnn_orig_truncatedloss_v22.Solver(objs, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=20000, labels=2,
                                                             shuffle=True, channels=numchannels, pad=pad,  textfn=textfn,
                                                             uncertainty=True, uncertfn= uncerts, channels2=3, valobj=valobjs, valuncertfn=valuncerts,
                                                                 numslices=numslices, slices=slices, axes=axes,
                                                                 dataset_portion=dataset_portion
                                                             )
        print('Starting second model training, iteration number {0}'.format(ITER + 1))
        # solver.train()
        solver.model.load_state_dict(torch.load(
            '/{3}/ref_e{0}_lr5e4_f{1}_checkpoint_model_{2}.pth'.format(epoch, numchannels,suffix,checkpoints_folder)))
        # torch.save(solver.model.state_dict(), '/{3}/ref_e{0}_lr5e4_f{1}_checkpoint_model_{2}.pth'.format(epoch, numchannels,suffix,checkpoints_folder))

        refineinterfeatimgs =[]
        refineoutputs = []
        for v in np.arange(len(objs)):
            refineinterfeatimg = '/{6}/{4}_{0}ch_refineinterfeatimg_added_e{1}_i{3}_{5}.nii.gz'.format(numchannels,
                                                                                                                 epoch, v, 0, subjs['train'][v],suffix,outputs_folder)
            refineoutput = '/{6}/{4}_{0}ch_refineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels, epoch, v, 0,
                                                                                                             subjs['train'][v],suffix,outputs_folder)
            refineinterfeatimgs.append(refineinterfeatimg)
            refineoutputs.append(refineoutput)

        label_OHE = solver.test(refineinterfeatimgs, refineoutputs, niis, batchsize=20000, imgs=objs, uncertfn=uncerts) #,
        del label_OHE
        del solver.data, solver.dataloader, solver.valdata, solver.valdataloader

        if validation:
            valrefineinterfeatimgs = []
            valrefineoutputs = []
            for i in np.arange(len(valobjs)):
                valrefineinterfeatimg = '/{6}/{4}_{0}ch_valrefineinterfeatimg_added_e{1}_i{3}_{5}.nii.gz'.format(
                    numchannels, epoch, i, 0,subjs['val'][i], suffix,outputs_folder)
                valrefineoutput = '/{6}/{4}_{0}ch_valrefineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(
                    numchannels, epoch, i, 0,subjs['val'][i], suffix,outputs_folder)
                valrefineinterfeatimgs.append(valrefineinterfeatimg)
                valrefineoutputs.append(valrefineoutput)

            label_OHE = solver.test(valrefineinterfeatimgs, valrefineoutputs, valniis, batchsize=20000, imgs=valobjs, uncertfn=valuncerts)
            del label_OHE
        ## time
        elapsed = time.time() - starttime
        print(elapsed/60)