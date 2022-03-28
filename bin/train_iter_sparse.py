
import nibabel as nib
import time
import numpy as np
import uncertainty
import torch
from datetime import datetime
from run import run_classify_weightedImg, run_two_stage_cnn_orig_truncatedloss_sparse
from preprocess import generate_wmsubgm_mask, generate_obj_files

## hyperparameters
numchannels = 4
iterations = 5
pad = 5
validation=True
iterative = 1
generator = False
epoch = 10
gm2wm = 0.91487 # ratio of number of voxels (gm to wm) [if skull stripping -> put brain:non-brain]
csf2wm = 0.48580 # ratio of number of voxels (csf to wm) [0 if skull stripping]
third_model = False # False if running mousepatch

## set file names and folder paths
# subjs = {'val':[ '056','010','115'], #
#         'train':['072','023','132' ] } #,'087',

subjs = {'val':[ '010'], #
        'train':['023'] } #,'087',

for numslices in [10]:
    print('Number of slices: {0}'.format(numslices))
# numslices = 10
    losses_folder = '/data/infant/losses/'
    checkpoints_folder = '/data/infant/checkpoints/'
    intermediate_folder = '/data/infant/intermediate_nii/'
    outputs_folder =  '/data/infant/outputs/'
    objects_folder = '/data/infant/objects/'
    fns = ['/{2}/{0}_p5_sparse_{1}slices.obj'.format(subjs['train'][i],numslices,objects_folder) for i in range(len(subjs['train']))]
    fns_whole = ['/{1}/{0}_N4_1mm_sparse.obj'.format(subjs['train'][i],objects_folder) for i in range(len(subjs['train']))]
    # fns = fns_whole
    valfns = ['/{2}/{0}_p5_sparse_{1}slices.obj'.format(subjs['val'][i],numslices,objects_folder) for i in range(len(subjs['val']))]
    valfns_whole = ['/{1}/{0}_N4_1mm_sparse.obj'.format(subjs['val'][i],objects_folder) for i in range(len(subjs['val']))]
    # valfns = valfns_whole
    masks = ['/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
    niis = [nib.load('/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subjs['train'][i]))._affine for i in range(len(subjs['train']))]
    valmasks = ['/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subjs['val'][i]) for i in range(len(subjs['val']))]
    valniis = [nib.load('/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subjs['val'][i]))._affine for i in range(len(subjs['val']))]
    labels = ['/data/infant/t2traindata_labels_handedit_YK_05072020/{0}-C-T1.T2w.final.label.nii.gz'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
    vallabels = ['/data/infant/t2traindata_labels_handedit_YK_05072020/{0}-C-T1.T2w.final.label.nii.gz'.format(subjs['val'][i]) for i in range(len(subjs['val']))]
    initlabels = ['/data/infant/outputs/023_4ch_refineoutput_added_e7_f1_i0_2021_50slices1.nii.gz']
    valinitlabels = ['/data/infant/outputs/010_4ch_valrefineoutput_added_e7_f1_i0_2021_50slices1.nii.gz']

    mean = []
    var = []
    valmean = []
    valvar = []

    starttime = time.time()
    for ITER in np.arange(iterative):
        print('Starting iteration number {0}'.format(ITER+1))
        # TODO: CHANGE SUFFIX ACCORDING TO RUN
        suffix = '2021_{0}slices_sparse'.format(numslices) + str(ITER+1)

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
            solver = run_two_stage_cnn_orig_truncatedloss_sparse.Solver(fns, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=5000,
                                                                        in_features=1, labels=3, shuffle=True, pad=pad, channels=1, textfn = textfn, suffix=suffix,
                                                                        valobj=valfns, spherecoord=False,
                                                                        initlabels=initlabels, valinitlabels=valinitlabels)
            print('Starting first model training, iteration number {0}, uncertainty iteration {1}'.format(ITER + 1, ii+1))
            solver.train()

            # solver.model.load_state_dict(torch.load(
            #     '/{4}/init_e{1}_lr5e4_f{0}_i{2}_{3}.pth'.format(numchannels,int(epoch),0,suffix),checkpoints_folder))
            # solver.train(epoch=epoch_ext)
            initinterfeatimgs = []
            initoutputs = []
            valinitinterfeatimgs = []
            valinitoutputs = []
            for i in np.arange(len(fns)):
                initinterfeatimg = '/{6}/{4}_{0}ch_initinterfeatimg_e{1}_i{3}_{5}.nii.gz'.format(numchannels, epoch, i, ii, subjs['train'][i], suffix,intermediate_folder)
                initoutput = '/{6}/{4}_{0}ch_initoutput_e{1}_i{3}_f{2}_{5}.nii.gz'.format(numchannels, epoch, i, ii, subjs['train'][i],suffix,intermediate_folder)
                initinterfeatimgs.append(initinterfeatimg)
                initoutputs.append(initoutput)
            if validation:
                for i in np.arange(len(valfns)):
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
                                       subjs['train'], numchannels, iterations,pkl=True, suffix=suffix)
        if validation:
            uncertainty.render_uncert_imgs(valfns_whole, valvar, valniis,
                                           subjs['val'], numchannels, iterations, pkl=True, suffix='val'+suffix)
        del var, valvar, mean, valmean

        #### Choose which model/intermediate dataset
        initinterfeatimgs_for_refstage, initoutputs_for_refstage, \
        valinitinterfeatimgs_for_refstage, valinitoutputs_for_refstage, smallestidx = uncertainty.select_best_model(gm2wm, csf2wm, iterations,
                                                        initoutputs_alliterations, initimodels, initinterfeatimgs_alliterations,
                                                        valinitinterfeatimgs, validation=True)
        print('First model finished. Generating pickled images, iteration number {0}'.format(ITER + 1))
        # smallestidx = 1
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
        for i in np.arange(len(fns)):
            uncertniis.append('/data/infant/variance/{0}_vars_i{2}_{1}ch_en_{3}_{4}.nii.gz'.format(subjs['train'][i], numchannels, iterations, i, suffix))
            obj = '/{6}/{4}_{0}ch_initinterfeatimg_e{1}_i{3}_{5}'.format(numchannels, epoch, i, smallestidx,
                                                                         subjs['train'][i], suffix, objects_folder)
            objs.append('{0}.obj'.format(obj))
            uncert = '//{6}/{4}_{0}ch_uncert_e{1}_f{2}_i{3}_{5}'.format(numchannels, epoch, i, smallestidx,
                                                                        subjs['train'][i], suffix, objects_folder)
            uncerts.append('{0}.obj'.format(uncert))
            objw = '//{6}/{4}_{0}ch_initinterfeatimg_e{1}_i{3}_{5}_whole'.format(numchannels, epoch, i, smallestidx,
                                                                                subjs['train'][i], suffix,
                                                                                objects_folder)
            objs_whole.append('{0}.obj'.format(objw))
            uncertw = '//{6}/{4}_{0}ch_uncert_e{1}_f{2}_i{3}_{5}_whole'.format(numchannels, epoch, i, smallestidx,
                                                                              subjs['train'][i], suffix, objects_folder)
            uncerts_whole.append('{0}.obj'.format(uncertw))


            fname = generate_obj_files.generate_mask(initinterfeatimgs_for_refstage[i].split('.')[0], fns[i])
            generate_obj_files.generate_obj_files(obj, initinterfeatimgs_for_refstage[i], fname, labels[i])
            generate_obj_files.generate_obj_files(uncert, uncertniis[i], fname, labels[i], numchannels=3)
            generate_obj_files.generate_obj_files(objw, initinterfeatimgs_for_refstage[i], masks[i], labels[i])
            generate_obj_files.generate_obj_files(uncertw, uncertniis[i], masks[i], labels[i], numchannels=3)
        if validation:
            for i in np.arange(len(valfns)):
                valuncertniis.append(
                    '//data/infant/variance/{0}_vars_i{2}_{1}ch_en_{3}_val{4}.nii.gz'.format(subjs['val'][i], numchannels, iterations, i,
                                                                                          suffix))
                obj = '//{6}/{4}_{0}ch_valinitinterfeatimg_e{1}_i{3}_{5}'.format(numchannels, epoch, i, smallestidx,
                                                                                 subjs['val'][i], suffix,
                                                                                 objects_folder)
                valobjs.append('{0}.obj'.format(obj))
                uncert = '//{6}/{4}_{0}ch_valuncert_e{1}_f{2}_i{3}_{5}'.format(numchannels, epoch, i, smallestidx,
                                                                               subjs['val'][i], suffix, objects_folder)
                valuncerts.append('{0}.obj'.format(uncert))
                valobj = '//{6}/{4}_{0}ch_valinitinterfeatimg_e{1}_i{3}_{5}_whole'.format(numchannels, epoch, i,
                                                                                       smallestidx, subjs['val'][i],
                                                                                       suffix, objects_folder)
                valobjs_whole.append('{0}.obj'.format(valobj))
                valuncert = '//{6}/{4}_{0}ch_valuncert_e{1}_f{2}_i{3}_{5}_whole'.format(numchannels, epoch, i, smallestidx,
                                                                                     subjs['val'][i], suffix,
                                                                                     objects_folder)
                valuncerts_whole.append('{0}.obj'.format(valuncert))



                fname = generate_obj_files.generate_mask(valinitinterfeatimgs_for_refstage[i].split('.')[0], valfns[i])
                generate_obj_files.generate_obj_files(obj, valinitinterfeatimgs_for_refstage[i], fname, vallabels[i])
                generate_obj_files.generate_obj_files(uncert, valuncertniis[i], fname, vallabels[i], numchannels=3)
                generate_obj_files.generate_obj_files(valobj, valinitinterfeatimgs_for_refstage[i], valmasks[i], vallabels[i])
                generate_obj_files.generate_obj_files(valuncert, valuncertniis[i], valmasks[i], vallabels[i], numchannels=3)

        ############################################################################################
        ############################################################################################
        ## Train the generator
        ############################################################################################
        ############################################################################################
        if generator:
            textfn = '/{1}/generator_train_losses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),losses_folder)
            with open(textfn, 'w') as f:
                f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
            solver = run_classify_weightedImg.Solver(fns, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=5000,
                                                     in_features=1, labels=3, shuffle=True, pad=pad,
                                                     channels=1, textfn = textfn, uncertainty=True, uncertfn= uncerts,
                                                     initmodel=initmodel, ae = False, valobj=valfns, valuncertfn=valuncerts)
            print('Starting generator training iteration number {0}'.format(ITER + 1))
            solver.train()

            # solver.model.load_state_dict(torch.load(
            #    '/{3}/ref_e{0}_lr5e4_f{1}_i{2}_checkpoint3.pth'.format(epoch2, numchannels, iterations,checkpoints_folder)))

            refineinterfeatimgs =[]
            refineoutputs = []
            for i in np.arange(len(objs)):
                refineinterfeatimg = '/{6}/{4}_{0}ch_modifiedinterfeatimg_e{1}_i{3}_{5}.nii.gz'.format(numchannels,
                                                                                                                     epoch, i, 0, subjs['train'][i],suffix,outputs_folder)
                refineoutput = '/{6}/{4}_{0}ch_modifiedoutput_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels, epoch, i, 0,
                                                                                                                 subjs['train'][i],suffix,outputs_folder)
                refineinterfeatimgs.append(refineinterfeatimg)
                refineoutputs.append(refineoutput)

            (label_OHE, fn_mods, fn_mod_adds) = solver.test(refineinterfeatimgs, refineoutputs, niis, batchsize=1000)

            if validation:
                valrefineinterfeatimgs = []
                valrefineoutputs = []
                for i in np.arange(len(valobjs)):
                    valrefineinterfeatimg = '/{6}/{4}_{0}ch_valmodifiedinterfeatimg_e{1}_i{3}_{5}.nii.gz'.format(
                        numchannels,epoch, i, 0, subjs['val'][i], suffix,outputs_folder)
                    valrefineoutput = '/{6}/{4}_{0}ch_valmodifiedoutput_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels,epoch, i, 0,
                                                                                                                     subjs['val'][i],suffix,outputs_folder)
                    valrefineinterfeatimgs.append(valrefineinterfeatimg)
                    valrefineoutputs.append(valrefineoutput)

                (_, _, val_fn_mod_adds) = solver.test(valrefineinterfeatimgs, valrefineoutputs, niis, batchsize=1000)

            fns = []
            valfns=[]
            print('Generating picked images for modified images, iteration number {0}'.format(ITER + 1))
            for i,fn_mod_add in enumerate(fn_mod_adds):
                prefix = fn_mod_add.split('.')[0]
                obj = prefix.split('modified_imgs')[0] + 'objects' + prefix.split('modified_imgs')[1]
                generate_obj_files.generate_obj_files(obj, fn_mod_add, masks[i], labels[i])
                fns.append(obj+'.obj')
            if validation:
                for i, val_fn_mod_add in enumerate(val_fn_mod_adds):
                    prefix = val_fn_mod_add.split('.')[0]
                    obj = prefix.split('modified_imgs')[0] + 'objects' + prefix.split('modified_imgs')[1]
                    generate_obj_files.generate_obj_files(obj, val_fn_mod_add, valmasks[i], vallabels[i])
                    valfns.append(obj+'.obj')

            torch.save(solver.modelWImg.state_dict(), '/data/infant/checkpoints/gen_e{0}_lr5e4_f{1}_checkpoint_modelWImg_{2}.pth'.format(epoch, numchannels,suffix))

        ############################################################################################
        ############################################################################################
        ### train the 2nd stage model
        ############################################################################################
        ############################################################################################
        # initoutputs_for_refstage = ['/data/infant/intermediate_nii/023_4ch_initoutput_e5_i1_f0_2021_10slices_sparse1.nii.gz']
        # valinitoutputs_for_refstage = ['/data/infant/intermediate_nii/010_4ch_valinitoutput_e5_i1_f0_2021_10slices_sparse1.nii.gz']
        textfn = '/{1}/secondmodel_train_losses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),losses_folder)

        # TODO: change init labels to reflect the output from the first model
        with open(textfn, 'w') as f:
            f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
        solver = run_two_stage_cnn_orig_truncatedloss_sparse.Solver(objs, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=1000, in_features=1, labels=3,
                                                                    shuffle=True, channels=numchannels, coords=False, DL=False, pad=pad, softdiceloss=False, textfn=textfn,
                                                                    uncertainty=True, uncertfn= uncerts, channels2=3, valobj=valobjs, valuncertfn=valuncerts, spherecoord=False,
                                                                    valinitlabels= valinitoutputs_for_refstage, initlabels= initoutputs_for_refstage)
        print('Starting second model training, iteration number {0}'.format(ITER + 1))
        solver.train()
        # solver.model.load_state_dict(torch.load(
        #     '/{3}/ref_e{0}_lr5e4_f{1}_i{2}_checkpoint3.pth'.format(epoch, numchannels, iterations,checkpoints_folder)))
        torch.save(solver.model.state_dict(), '/{3}/ref_e{0}_lr5e4_f{1}_checkpoint_model_{2}.pth'.format(epoch, numchannels,suffix,checkpoints_folder))

        refineinterfeatimgs =[]
        refineoutputs = []
        for v in np.arange(len(objs)):
            refineinterfeatimg = '/{6}/{4}_{0}ch_refineinterfeatimg_added_e{1}_i{3}_{5}.nii.gz'.format(numchannels,
                                                                                                                 epoch, v, 0, subjs['train'][v],suffix,outputs_folder)
            refineoutput = '/{6}/{4}_{0}ch_refineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels, epoch, v, 0,
                                                                                                             subjs['train'][v],suffix,outputs_folder)
            refineinterfeatimgs.append(refineinterfeatimg)
            refineoutputs.append(refineoutput)

        label_OHE = solver.test(refineinterfeatimgs, refineoutputs, niis, batchsize=1000, imgs=objs_whole, uncertfn=uncerts_whole)
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

            label_OHE = solver.test(valrefineinterfeatimgs, valrefineoutputs, valniis, batchsize=1000, imgs=valobjs_whole, uncertfn=valuncerts_whole)
            del label_OHE
        ## time
        elapsed = time.time() - starttime
        print(elapsed/60)

        if third_model:
            ############################################################################################
            ############################################################################################
            ### generate obj files
            ############################################################################################
            ############################################################################################
            objs=[]
            valobjs=[]
            # refineinterfeatimgs = ['/data/infant/outputs/072_4ch_refineinterfeatimg_added_e7_fn0_i0_2021_25slices_rerun1.nii.gz',
            #                        '/data/infant/outputs/023_4ch_refineinterfeatimg_added_e7_fn1_i0_2021_25slices_rerun1.nii.gz',
            #                        '/data/infant/outputs/132_4ch_refineinterfeatimg_added_e7_fn2_i0_2021_25slices_rerun1.nii.gz']
            # valrefineinterfeatimgs = ['/data/infant/outputs/056_4ch_valrefineinterfeatimg_added_e7_fn0_i0_2021_25slices_rerun1.nii.gz',
            #                           '/data/infant/outputs/010_4ch_valrefineinterfeatimg_added_e7_fn1_i0_2021_25slices_rerun1.nii.gz',
            #                           '/data/infant/outputs/115_4ch_valrefineinterfeatimg_added_e7_fn2_i0_2021_25slices_rerun1.nii.gz']

            for i in np.arange(0,len(refineinterfeatimgs)):
                prefix = '/data/infant/T2_cerebrum_for_generate_wmsubgm/{0}-C-T1_T2w.1mm.N4.cerebrum'.format(subjs['train'][i])
                generate_wmsubgm_mask.generate_wmsubgm_mask(refineoutputs[i], prefix, masks[i])
                # fname = '/data/infant/intermediate_nii/{0}_4ch_refineoutput_added_e7_f{1}_i0_2021_25slices_rerun1.masked.morph.mask.nii.gz'.format(subjs['train'][i],i)
                obj = '/{1}/{0}'.format(refineinterfeatimgs[i].split('.')[0].split('/')[-1],objects_folder)
                objs.append('{0}.obj'.format(obj))
                generate_obj_files.generate_obj_files(obj, refineinterfeatimgs[i], prefix + '.masked.inters.mask.nii.gz', labels[i], numchannels=4)

            if validation:
                for i in np.arange(0, len(valrefineinterfeatimgs)):
                    prefix = '/data/infant/T2_cerebrum_for_generate_wmsubgm/{0}-C-T1_T2w.1mm.N4.cerebrum'.format(
                        subjs['val'][i])
                    generate_wmsubgm_mask.generate_wmsubgm_mask(valrefineoutputs[i], prefix, valmasks[i])
                    # fname = '/data/infant/intermediate_nii/{0}_4ch_valrefineoutput_added_e7_f{1}_i0_2021_25slices_rerun1.masked.morph.mask.nii.gz'.format(
                    #     subjs['val'][i], i)
                    obj = '/{1}/{0}'.format(valrefineinterfeatimgs[i].split('.')[0].split('/')[-1],objects_folder)
                    valobjs.append('{0}.obj'.format(obj))
                    generate_obj_files.generate_obj_files(obj, valrefineinterfeatimgs[i], prefix + '.masked.inters.mask.nii.gz', vallabels[i], numchannels=4)

                    ## generate pickled whole images
                    # if numslices is not None:
                    #     valobjs_whole = []
                    #     for i in np.arange(0, len(valrefineinterfeatimgs)):
                    #         obj = '//{6}/{4}_{0}ch_valrefineinterfeatimg_e{1}_i{3}_{5}_whole'.format(numchannels,epoch, i,
                    #                                                                                                iterations,subjs['val'][i],suffix,objects_folder)
                    #         valobjs_whole.append('{0}.obj'.format(obj))
                    #         generate_obj_files.generate_obj_files(obj, valrefineinterfeatimgs[i], valmasks[i],vaGllabels[i])


            ## time
            elapsed = time.time() - starttime
            print(elapsed/60)

