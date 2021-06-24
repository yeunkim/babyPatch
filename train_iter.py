
import pickle
import nibabel as nib
import time
import numpy as np
import data_preproc_noupsample
import uncertainty
import torch
from datetime import datetime
import run_classify_weightedImg
import run_two_stage_cnn_orig_truncatedloss
import data_preproc_onestage
import generate_obj_files

numchannels = 4
iterations = 2
pad = 5

subjs = {'val':[ '056'], #, '010','115'
        'train':['072'] } #,,'087',,,'023' ,'132'

fns = ['/data/infant/objects/{0}_p5_10slices.obj'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
fns_whole = ['/data/infant/objects/{0}_N4_1mm.obj'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
valfns = ['/data/infant/objects/{0}_p5_10slices.obj'.format(subjs['val'][i]) for i in range(len(subjs['val']))]
valfns_whole = ['/data/infant/objects/{0}_N4_1mm.obj'.format(subjs['val'][i]) for i in range(len(subjs['val']))]
masks = ['/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
niis = [nib.load('/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subjs['train'][i]))._affine for i in range(len(subjs['train']))]
valmasks = ['/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subjs['val'][i]) for i in range(len(subjs['val']))]
valniis = [nib.load('/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subjs['val'][i]))._affine for i in range(len(subjs['val']))]
labels = ['/data/infant/t2traindata_labels_handedit_YK_05072020/{0}-C-T1.T2w.final.label.nii.gz'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
vallabels = ['/data/infant/t2traindata_labels_handedit_YK_05072020/{0}-C-T1.T2w.final.label.nii.gz'.format(subjs['val'][i]) for i in range(len(subjs['val']))]

starttime = time.time()
generator = False

epoch = 2

spherecoords=[]
# subjs=[ '087-C-T1', '056-C-T1',  '002-C-T1', '072-C-T1', '108-C-T1']
# for i in range(len(subjs)):
#     fns.append('/data/infant/objects/{0}_T2w_p6_spherecoords_pruned.obj'.format(subjs[i]))
#     valfns.append('/data/infant/objects/{0}_T2w_p6_spherecoords.obj'.format(subjs[i]))
#     spherecoords.append(
#         '/data/infant/spherecoords/atlas_Int_M6_T2_cartesian_new_coords_masked_{0}_xfmed_spherecoord2_filled.nii.gz'.format(subjs[i]))
#     if subjs[i] != '072-C-T1':
#         masks.append('/nafs/shattuck/yeunkim/infant_images/rebeccabelisle/{0}_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subjs[i]))
#     else:
#         masks.append('//oldmiro/data/SSD_data/infant/processed/train_data/{0}/{0}_T2w.1mm.N4.subcort.mask.nii.gz'.format('072'))
#     labels.append('/data/infant/labels/{0}_T2w.label.nii.gz'.format(subjs[i]))
#     if subjs[i] != '108-C-T1':
#         niis.append(nib.load('/nafs/shattuck/yeunkim/infant_images/rebeccabelisle/{0}_T2w.1mm.N4.cerebrum.nii.gz'.format(subjs[i]))._affine)
#     else:
#         niis.append(nib.load('/data/infant/T2_2019/108-C-T1_T2w.1mm.N4.cerebrum.bfc.nii.gz')._affine)

# valspherecoords=[]
# valspherecoords.append('/data/infant/spherecoords/atlas_Int_M6_T2_cartesian_new_coords_masked_{0}_xfmed_spherecoord_filled.nii.gz'.format(valsubjs[0]))


mean = []
var = []
valmean = []
valvar = []

validation=True
iterative = 1

for ITER in np.arange(iterative):
    print('Starting iteration number {0}'.format(ITER+1))
    suffix = '2021_10slices' + str(ITER)

    ############################################################################################
    ############################################################################################
    ### train the 1st model
    ############################################################################################
    ############################################################################################
    initinterfeatimgs_alliterations = []
    initoutputs_alliterations = []
    initimodels = []
    for ii in np.arange(iterations):

        textfn = '/data/infant/losses/avglosses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'))
        with open(textfn, 'w') as f:
            f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
        solver = run_two_stage_cnn_orig_truncatedloss.Solver(fns, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=5000,
                                             in_features=1, labels=3, shuffle=True, pad=pad,
                                             channels=1, textfn = textfn, suffix=suffix,
                                                             valobj=valfns,spherecoord=False
                                                             )
        print('Starting first model training, iteration number {0}, uncertainty iteration {1}'.format(ITER + 1, ii+1))
        solver.train()

        # solver.model.load_state_dict(torch.load(
        #     '/data/infant/checkpoints/init_e{1}_lr5e4_f{0}_i{2}_{3}.pth'.format(numchannels, int(epoch), 0,suffix)))
        # solver.train(epoch=epoch_ext)
        initinterfeatimgs = []
        initoutputs = []
        valinitinterfeatimgs = []
        valinitoutputs = []
        for i in np.arange(len(fns)):
            initinterfeatimg = '//data/infant/intermediate_nii/{4}_{0}ch_initinterfeatimg_e{1}_i{3}_fn{2}_{5}.nii.gz'.format(numchannels, epoch, i, ii, subjs['train'][i], suffix)
            initoutput = '//data/infant/intermediate_nii/{4}_{0}ch_initoutput_e{1}_i{3}_f{2}_{5}.nii.gz'.format(numchannels, epoch, i, ii, subjs['train'][i],suffix)
            initinterfeatimgs.append(initinterfeatimg)
            initoutputs.append(initoutput)
        if validation:
            for i in np.arange(len(valfns)):
                valinitinterfeatimg = '//data/infant/intermediate_nii/{4}_{0}ch_valinitinterfeatimg_e{1}_i{3}_fn{2}_{5}.nii.gz'.format(
                    numchannels, epoch, i, ii, subjs['val'][i], suffix)
                valinitoutput = '//data/infant/intermediate_nii/{4}_{0}ch_valinitoutput_e{1}_i{3}_f{2}_{5}.nii.gz'.format(
                    numchannels, epoch, i, ii,subjs['val'][i], suffix)
                valinitinterfeatimgs.append(valinitinterfeatimg)
                valinitoutputs.append(valinitoutput)
            initinterfeatimgs_alliterations.append(initinterfeatimgs + valinitinterfeatimgs)
            initoutputs_alliterations.append(initoutputs + valinitoutputs)
        else:
            initinterfeatimgs_alliterations.append(initinterfeatimgs)
            initoutputs_alliterations.append(initoutputs)
        initmodel = ('/data/infant/checkpoints/init_e{1}_lr5e4_f{0}_i{2}_{3}.pth'.format(numchannels, int(epoch), ii,
                                                                                         suffix))
        torch.save(solver.model.state_dict(), initmodel)
        initimodels.append(initmodel)


        label_OHE = solver.test(initinterfeatimgs, initoutputs, niis,batchsize=5000, imgs=fns_whole)

        mean, var = uncertainty.compute_var_mean(label_OHE, mean, var, ii)
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
    gm2wm = 0.91487
    csf2wm = 0.48580
    initinterfeatimgs_for_refstage, initoutputs_for_refstage, \
    valinitinterfeatimgs_for_refstage, valinitoutputs_for_refstage, smallestidx = uncertainty.select_best_model(gm2wm, csf2wm, iterations,
                                                    initoutputs_alliterations, initimodels, initinterfeatimgs_alliterations,
                                                    valinitinterfeatimgs, validation=True)

    ############################################################################################
    ############################################################################################
    ### generate obj files
    ############################################################################################
    ############################################################################################
    uncertniis = []
    valuncertniis = []
    numslices = 10
    objs = []
    uncerts = []
    valobjs = []
    valuncerts = []
    for i in np.arange(len(fns)):
        uncertniis.append('/data/infant/variance/{0}_vars_i{2}_{1}ch_en_{3}_{4}.nii.gz'.format(subjs['train'][i], numchannels, iterations, i, suffix))
        fname = generate_obj_files.generate_mask(initinterfeatimgs_for_refstage[i])
        obj = '/data/infant/objects/{4}_{0}ch_initinterfeatimg_e{1}_fn{2}_i{3}_{5}'.format(numchannels, epoch, i, smallestidx,
                                                                                            subjs['train'][i], suffix)
        objs.append('{0}.obj'.format(obj))
        generate_obj_files.generate_obj_files(obj,initinterfeatimgs_for_refstage[i],fname,labels[i])
        uncert = '//data/infant/objects/{4}_{0}ch_uncert_e{1}_f{2}_i{3}_{5}'.format(numchannels, epoch, i, smallestidx,
                                                                                    subjs['train'][i], suffix)
        uncerts.append('{0}.obj'.format(uncert))
        generate_obj_files.generate_obj_files(uncert, uncertniis[i], fname, labels[i])
        ## generate whole images
        objs_whole = []
        uncerts_whole = []
        obj = '//data/infant/objects/{4}_{0}ch_initinterfeatimg_e{1}_fn{2}_i{3}_{5}_whole'.format(numchannels, epoch, i,smallestidx,
                                                                                                  subjs['train'][i],suffix)
        objs_whole.append('{0}.obj'.format(obj))
        generate_obj_files.generate_obj_files(obj, initinterfeatimgs_for_refstage[i], masks[i], labels[i])
        uncert = '//data/infant/objects/{4}_{0}ch_uncert_e{1}_f{2}_i{3}_{5}_whole'.format(numchannels, epoch, i,smallestidx,
                                                                                          subjs['train'][i],suffix)
        uncerts_whole.append('{0}.obj'.format(uncert))
        generate_obj_files.generate_obj_files(uncert, uncertniis[i], masks[i], labels[i])
    if validation:
        for i in np.arange(len(valfns)):
            valuncertniis.append(
                '//data/infant/variance/{0}_vars_i{2}_{1}ch_en_{3}_val{4}.nii.gz'.format(subjs['val'][i], numchannels, iterations, i,
                                                                                      suffix))
            fname = generate_obj_files.generate_mask(valinitinterfeatimgs_for_refstage[i])
            obj = '//data/infant/objects/{4}_{0}ch_valinitinterfeatimg_e{1}_fn{2}_i{3}_{5}'.format(numchannels, epoch,i, smallestidx,
                                                                                                   subjs['val'][i], suffix)
            valobjs.append('{0}.obj'.format(obj))
            generate_obj_files.generate_obj_files(obj, valinitinterfeatimgs_for_refstage[i], fname, vallabels[i])
            uncert = '//data/infant/objects/{4}_{0}ch_valuncert_e{1}_f{2}_i{3}_{5}'.format(numchannels, epoch, i,smallestidx,
                                                                                           subjs['val'][i], suffix)
            valuncerts.append('{0}.obj'.format(uncert))
            generate_obj_files.generate_obj_files(uncert, valuncertniis[i], fname, vallabels[i])
            ## generate whole images
            valobjs_whole = []
            valuncerts_whole = []
            obj = '//data/infant/objects/{4}_{0}ch_valinitinterfeatimg_e{1}_fn{2}_i{3}_{5}_whole'.format(numchannels, epoch, i,
                                                                                                         iterations,subjs['val'][i],suffix)
            valobjs_whole.append('{0}.obj'.format(obj))
            generate_obj_files.generate_obj_files(obj, valinitinterfeatimgs_for_refstage[i], valmasks[i], vallabels[i])
            uncert = '//data/infant/objects/{4}_{0}ch_valuncert_e{1}_f{2}_i{3}_{5}_whole'.format(numchannels, epoch,i, iterations,
                                                                                                 subjs['val'][i],suffix)
            valuncerts_whole.append('{0}.obj'.format(uncert))
            generate_obj_files.generate_obj_files(uncert, valuncertniis[i], valmasks[i], vallabels[i])

    print('First model finished. Generating pickled images, iteration number {0}'.format(ITER + 1))


    ############################################################################################
    ############################################################################################
    ## Train the generator
    ############################################################################################
    ############################################################################################
    if generator:
        textfn = '/data/infant/losses/avglosses_generator_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'))
        with open(textfn, 'w') as f:
            f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
        solver = run_classify_weightedImg.Solver(fns, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=5000,
                                             in_features=1, labels=3, shuffle=True, pad=pad,
                                             channels=1, textfn = textfn,uncertainty=True, uncertfn= uncerts,
                                                 initmodel=initmodel, ae = False, valobj=valfns, valuncertfn=valuncerts)
        print('Starting generator training iteration number {0}'.format(ITER + 1))
        solver.train()

        # solver.model.load_state_dict(torch.load(
        #    '/mnt/data/infant/checkpoints/ref_e{0}_lr5e4_f{1}_i{2}_checkpoint3.pth'.format(epoch2, numchannels, iterations)))

        refineinterfeatimgs =[]
        refineoutputs = []
        for i in np.arange(len(objs)):
            refineinterfeatimg = '//data/infant/outputs/{4}_{0}ch_modifiedinterfeatimg_e{1}_fn{2}_i{3}_{5}.nii.gz'.format(numchannels,
                                                                                                                 epoch, i, 0,
                                                                                                                          subjs['train'][i],suffix)
            refineoutput = '//data/infant/outputs/{4}_{0}ch_modifiedoutput_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels, epoch, i, 0,
                                                                                                             subjs['train'][i],suffix)
            refineinterfeatimgs.append(refineinterfeatimg)
            refineoutputs.append(refineoutput)

        (label_OHE, fn_mods, fn_mod_adds) = solver.test(refineinterfeatimgs, refineoutputs, niis, batchsize=1000)

        if validation:
            valrefineinterfeatimgs = []
            valrefineoutputs = []
            for i in np.arange(len(valobjs)):
                valrefineinterfeatimg = '//data/infant/outputs/{4}_{0}ch_valmodifiedinterfeatimg_e{1}_fn{2}_i{3}_{5}.nii.gz'.format(
                    numchannels,
                    epoch, i, 0,
                    subjs['val'][i], suffix)
                valrefineoutput = '//data/infant/outputs/{4}_{0}ch_valmodifiedoutput_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels,epoch, i, 0,
                                                                                                                 subjs['val'][i],suffix)
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
    textfn = '/data/infant/losses/avglosses_ref_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'))

    with open(textfn, 'w') as f:
        f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
    solver = run_two_stage_cnn_orig_truncatedloss.Solver(objs,  epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=1000, in_features=1, labels=3,
                                      shuffle=True, channels=numchannels,coords=False, DL=False, pad=pad, softdiceloss=False,
                                      uncertainty=True, uncertfn= uncerts, channels2=3, valobj=valobjs, valuncertfn=valuncerts, spherecoord=False
                                         )
    print('Starting second model training, iteration number {0}'.format(ITER + 1))
    solver.train()
    # solver.model.load_state_dict(torch.load(
    #     '/mnt/data/infant/checkpoints/ref_e{0}_lr5e4_f{1}_i{2}_checkpoint3.pth'.format(epoch, numchannels, iterations)))
    torch.save(solver.model.state_dict(), '/data/infant/checkpoints/ref_e{0}_lr5e4_f{1}_checkpoint_model_{2}.pth'.format(epoch, numchannels,suffix))

    refineinterfeatimgs =[]
    refineoutputs = []
    for i in np.arange(len(objs)):
        refineinterfeatimg = '//data/infant/outputs/{4}_{0}ch_refineinterfeatimg_added_e{1}_fn{2}_i{3}_{5}.nii.gz'.format(numchannels,
                                                                                                             epoch, i, 0, subjs['train'][i],suffix)
        refineoutput = '//data/infant/outputs/{4}_{0}ch_refineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels, epoch, i, 0,
                                                                                                         subjs['train'][i],suffix)
        refineinterfeatimgs.append(refineinterfeatimg)
        refineoutputs.append(refineoutput)

    label_OHE = solver.test(refineinterfeatimgs, refineoutputs, niis, batchsize=1000, imgs=objs_whole, uncertfn=uncerts_whole)
    del label_OHE

    if validation:
        valrefineinterfeatimgs = []
        valrefineoutputs = []
        for i in np.arange(len(valobjs)):
            valrefineinterfeatimg = '//data/infant/outputs/{4}_{0}ch_valrefineinterfeatimg_added_e{1}_fn{2}_i{3}_{5}.nii.gz'.format(
                numchannels,
                epoch, i, 0,
                subjs['val'][i], suffix)
            valrefineoutput = '//data/infant/outputs/{4}_{0}ch_valrefineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(
                numchannels, epoch, i, 0,
                subjs['val'][i], suffix)
            valrefineinterfeatimgs.append(valrefineinterfeatimg)
            valrefineoutputs.append(valrefineoutput)

        label_OHE = solver.test(valrefineinterfeatimgs, valrefineoutputs, niis, batchsize=1000, imgs=valobjs_whole, uncertfn=valuncerts_whole)

    ## time
    elapsed = time.time() - starttime
    print(elapsed/60)


    ############################################################################################
    ############################################################################################
    ### generate obj files
    ############################################################################################
    ############################################################################################
    objs=[]
    valobjs=[]

    for i in np.arange(0,len(refineinterfeatimgs)):
        fname = initinterfeatimgs_for_refstage[i].split('.')[0] + '_mask.nii.gz'
        obj = '/data/infant/objects/{0}.obj'.format(refineinterfeatimgs[i].split('.')[0].split('/')[-1])
        objs.append(obj)
        generate_obj_files.generate_obj_files(obj, refineinterfeatimgs[i], fname, labels[i])

    if validation:
        for i in np.arange(0, len(valrefineinterfeatimgs)):
            fname = valinitinterfeatimgs_for_refstage[i].split('.')[0] + '_mask.nii.gz'
            obj = '/data/infant/objects/{0}.obj'.format(valrefineinterfeatimgs[i].split('.')[0].split('/')[-1])
            valobjs.append(obj)
            generate_obj_files.generate_obj_files(obj, valrefineinterfeatimgs[i], fname, vallabels[i])

            ## generate pickled whole images
            if numslices is not None:
                valobjs_whole = []
                for i in np.arange(0, len(valrefineinterfeatimgs)):
                    obj = '//data/infant/objects/{4}_{0}ch_valrefineinterfeatimg_e{1}_fn{2}_i{3}_{5}_whole'.format(numchannels,epoch, i,
                                                                                                           iterations,subjs['val'][i],suffix)
                    valobjs_whole.append('{0}.obj'.format(obj))
                    generate_obj_files.generate_obj_files(obj, valrefineinterfeatimgs[i], valmasks[i],vallabels[i])

    ############################################################################################
    ############################################################################################
    ### train the 3rd stage model
    ############################################################################################
    ############################################################################################
    textfn = '/data/infant/losses/avglosses_ref_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'))

    with open(textfn, 'w') as f:
        f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
    solver = run_two_stage_cnn_orig_truncatedloss.Solver(objs,  epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=1000, in_features=1, labels=3,
                                      shuffle=True, channels=numchannels,coords=False, DL=False, pad=pad, softdiceloss=False,
                                      uncertainty=False, valobj=valobjs, spherecoord=False
                                         )
    print('Starting third model training, iteration number {0}'.format(ITER + 1))
    solver.train()
    # solver.model.load_state_dict(torch.load(
    #     '/mnt/data/infant/checkpoints/ref_e{0}_lr5e4_f{1}_i{2}_checkpoint3.pth'.format(epoch, numchannels, iterations)))
    torch.save(solver.model.state_dict(), '/data/infant/checkpoints/ref2_e{0}_lr5e4_f{1}_checkpoint_model_{2}.pth'.format(epoch, numchannels,suffix))

    refineinterfeatimgs =[]
    refineoutputs = []
    for i in np.arange(len(objs)):
        refineinterfeatimg = '//data/infant/outputs/{4}_{0}ch_refine2interfeatimg_added_e{1}_fn{2}_i{3}_{5}.nii.gz'.format(numchannels,
                                                                                                             epoch, i, 0,
                                                                                                                      subjs['train'][i],suffix)
        refineoutput = '//data/infant/outputs/{4}_{0}ch_refine2output_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels, epoch, i, 0,
                                                                                                         subjs['train'][i],suffix)
        refineinterfeatimgs.append(refineinterfeatimg)
        refineoutputs.append(refineoutput)

    label_OHE = solver.test(refineinterfeatimgs, refineoutputs, niis, batchsize=1000)
    del label_OHE

    if validation:
        valrefineinterfeatimgs = []
        valrefineoutputs = []
        for i in np.arange(len(valobjs)):
            valrefineinterfeatimg = '//data/infant/outputs/{4}_{0}ch_valrefine2interfeatimg_added_e{1}_fn{2}_i{3}_{5}.nii.gz'.format(
                numchannels,
                epoch, i, 0,
                subjs['val'][i], suffix)
            valrefineoutput = '//data/infant/outputs/{4}_{0}ch_valrefine2output_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(
                numchannels, epoch, i, 0,
                subjs['val'][i], suffix)
            valrefineinterfeatimgs.append(valrefineinterfeatimg)
            valrefineoutputs.append(valrefineoutput)

        label_OHE = solver.test(valrefineinterfeatimgs, valrefineoutputs, niis, batchsize=1000, imgs=valobjs_whole)

    ## time
    elapsed = time.time() - starttime
    print(elapsed/60)









## evaluate accuracy for # of slices
from diceCoeff import diceCoeff
slices = [15, 20,25]

# print('First model results:')
# for ss in slices:
#     print('{0} slices.'.format(ss))
#     for id in range(0,len(subjs['train'])):
#         print("Train data:")
#         nii = nib.load(initoutputs_for_refstage[id])
#         gt = nib.load(labels[id])
#         mask = nib.load(masks[id])
#         data = nii.get_fdata(nii)
#         datagt = nii.get_fdata(gt)
#         datamask = nii.get_fdata(mask)
#         data[datamask ==0] =0
#         datagt[datamask == 0] = 0
#         print(initoutputs_for_refstage[id])
#         dc_wm = diceCoeff(data, datagt, 1)
#         dc_gm = diceCoeff(data, datagt, 2)
#         dc_csf = diceCoeff(data, datagt, 3)
#         print("wm: {0}  gm: {1}  csf: {2}  ".format(dc_wm, dc_gm, dc_csf))
#     for id in range(0,len(subjs['val'])):
#         print("Validation data:")
#         nii = nib.load(valinitoutputs_for_refstage[id])
#         gt = nib.load(vallabels[id])
#         mask = nib.load(valmasks[id])
#         data = nii.get_fdata(nii)
#         datagt = nii.get_fdata(gt)
#         datamask = nii.get_fdata(mask)
#         data[datamask ==0] =0
#         datagt[datamask == 0] = 0
#         print(valinitoutputs_for_refstage[id])
#         dc_wm = diceCoeff(data, datagt, 1)
#         dc_gm = diceCoeff(data, datagt, 2)
#         dc_csf = diceCoeff(data, datagt, 3)
#         print("wm: {0}  gm: {1}  csf: {2}  ".format(dc_wm, dc_gm, dc_csf))

print('Second model results:')
for ss in slices:
    print('{0} slices.'.format(ss))
    suffix = '2021_{0}slices'.format(ss) + str(0)
    refineoutputs = []
    for i in np.arange(3):
        refineoutput = '//data/infant/outputs/{4}_{0}ch_refineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(
            numchannels, epoch, i, 0,
            subjs['train'][i], suffix)
        refineoutputs.append(refineoutput)
    valrefineoutputs = []
    for i in np.arange(3):
        valrefineoutput = '//data/infant/outputs/{4}_{0}ch_valrefineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(
            numchannels, epoch, i, 0,
            subjs['val'][i], suffix)
        valrefineoutputs.append(valrefineoutput)
    # for id in range(0,len(subjs['train'])):
    #     print("Train data:")
    #     nii = nib.load(refineoutputs[id])
    #     gt = nib.load(labels[id])
    #     mask = nib.load(masks[id])
    #     data = nii.get_fdata()
    #     datagt = gt.get_fdata()
    #     datamask = mask.get_fdata()
    #     data[datamask ==0] =0
    #     datagt[datamask == 0] = 0
    #     print(refineoutputs[id])
    #     dc_wm = diceCoeff(data, datagt, 1)
    #     dc_gm = diceCoeff(data, datagt, 2)
    #     dc_csf = diceCoeff(data, datagt, 3)
    #     print("wm: {0}  gm: {1}  csf: {2}  ".format(dc_wm, dc_gm, dc_csf))
    for id in range(0,len(subjs['val'])):
        print("Validation data:")
        nii = nib.load(valrefineoutputs[id])
        gt = nib.load(vallabels[id])
        mask = nib.load(valmasks[id])
        data = nii.get_fdata()
        datagt = gt.get_fdata()
        datamask = mask.get_fdata()
        data[datamask ==0] =0
        datagt[datamask == 0] = 0
        print(valrefineoutputs[id])
        dc_wm = diceCoeff(data, datagt, 1)
        dc_gm = diceCoeff(data, datagt, 2)
        dc_csf = diceCoeff(data, datagt, 3)
        print("wm: {0}  gm: {1}  csf: {2}  ".format(dc_wm, dc_gm, dc_csf))