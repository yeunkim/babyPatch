# skull-stripping


import nibabel as nib
import time
import numpy as np
import uncertainty
import torch
from datetime import datetime
from run import run_two_stage_cnn_orig_truncatedloss_v22
from preprocess import generate_wmsubgm_mask, generate_obj_files
from preprocess import data_preproc_v22

## hyperparameters
numchannels = 4
iterations = 1
pad = 10
validation=True
iterative = 1
generator = False
epoch = 10
gm2wm = 0.5 # ratio of number of voxels (gm to wm) [if skull stripping -> put brain:non-brain]
csf2wm = 0 # ratio of number of voxels (csf to wm) [0 if skull stripping]
third_model = False
numslices =(60,-1,60)
slices = True
axes = (True,True,True)
dataset_portion = 0.50
pkl = False
planes = (True, False, True)

species = 'rat'
## set file names and folder paths
# subjs = {'val':['inj_050_12d','shm_030_30d'], #
#         'train':[# "inj_025_24h",
#                 "inj_013_24h",
#                 "shm_018_24h",
#                 'inj_023_12d',
#                 # "shm_036_24h",
#                 # "inj_050_12d",
#                 "shm_027_12d",
#                 "shm_015_12d",
#                 "inj_008_30d"
#                 # "inj_038_30d",
#                 # "shm_033_30d"
#         ]}
subjs = {'val': [
                 ],
         'train': [#'shm_015_12d'
                'inj_023_12d','shm_027_12d'
                   ]}

datadir='/home/lucydunnlocal/Final_Train12/Final_Train12/'
# datadir = '/allMouseTrainData/'
for slice_num in [10]:
    print('Slice interval: {0}'.format(slice_num))
    print('Dataset portion: {0}'.format(dataset_portion))
    losses_folder = '/data/{0}/losses/'.format(species)
    checkpoints_folder = '/data/{0}/checkpoints/'.format(species)
    intermediate_folder = '/data/{0}/intermediate_nii/'.format(species)
    outputs_folder = '/data/{0}/outputs/'.format(species)
    objects_folder = '/data/{0}/objects/'.format(species)
    fns_whole = ['/{1}/{0}_p10_xz.h5'.format(subjs['train'][i],objects_folder) for i in range(len(subjs['train']))]
    # niis = [nib.load('/data/{1}/{2}/{0}/brain.reorient.resampled0.1mm.int16.label.nii.gz'.format(subjs['train'][i], species, datadir))._affine for i in range(len(subjs['train']))]
    # labels = ['/data/{1}/{2}/{0}/brain.reorient.resampled0.1mm.int16.label.nii.gz'.format(subjs['train'][i], species, datadir) for i in range(len(subjs['train']))]
    # data = ['/data/{1}/{2}/{0}/anat.reorient.resampled0.1mm.nii.gz'.format(subjs['train'][i],species, datadir) for i in range(len(subjs['train']))]
    data = ['/{0}/{1}/2T2star_MEAN.nii.gz'.format(datadir, subjs['train'][i]) for i in
            range(len(subjs['train']))]
    niis = [nib.load('/{0}/{1}/2T2star_MEAN.nii.gz'.format(datadir, subjs['train'][i]))._affine for i in
            range(len(subjs['train']))]
    mean = []
    var = []

    starttime = time.time()
    for ITER in np.arange(iterative):
        print('Starting iteration number {0}'.format(ITER+1))
        suffix = '2022_{0}slices'.format(slice_num) + str(ITER+1)

        ############################################################################################
        ############################################################################################
        ### train the 1st model
        ############################################################################################
        ############################################################################################
        initinterfeatimgs_alliterations = []
        initoutputs_alliterations = []
        initimodels = []
        for ii in np.arange(4,5):
            label_OHEs = []
            textfn = '/{1}/avglosses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),losses_folder)
            with open(textfn, 'w') as f:
                f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
            print('Starting first model training, iteration number {0}, uncertainty iteration {1}'.format(ITER + 1, ii+1))
            solver = run_two_stage_cnn_orig_truncatedloss_v22.Solver(fns_whole, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=3000,
                                            labels=2, shuffle=True, pad=pad, channels=1, textfn = textfn, suffix=suffix,
                                            numslices= numslices, slices=slices, axes = axes, dataset_portion=dataset_portion, twoplane=planes
                                            )
            # initmodel = ('/{4}/init_e{1}_f{0}_i{2}_{3}.pth'.format(numchannels,int(epoch),ii,suffix,checkpoints_folder))
            initmodel = '/data/{0}/checkpoints/'.format('mouse') + '/init_e5_f4_i3_2022_10slices1.pth'

            solver.model.load_state_dict(torch.load(initmodel))
            initinterfeatimgs = []
            initoutputs = []
            for i in np.arange(len(fns_whole)):
                initinterfeatimg = '/{6}/{4}_{0}ch_initinterfeatimg_e{1}_i{3}_{5}.nii.gz'.format(numchannels, epoch, i, ii, subjs['train'][i], suffix,intermediate_folder)
                initoutput = '/{6}/{4}_{0}ch_initoutput_e{1}_i{3}_f{2}_{5}.nii.gz'.format(numchannels, epoch, i, ii, subjs['train'][i],suffix,intermediate_folder)
                initinterfeatimgs.append(initinterfeatimg)
                initoutputs.append(initoutput)
                initinterfeatimgs_alliterations.append(initinterfeatimgs)
                initoutputs_alliterations.append(initoutputs)

            for TD in range(0,len(fns_whole)):
                print(data[TD])
                datastruct = data_preproc_v22.imagepatches(fname=data[TD], num_classes=2, pad=pad, masklabel=False, ram=True)
                dataset = datastruct.return_data_struct()
                del datastruct
                label_OHE = solver.test(initinterfeatimgs[TD], initoutputs[TD], niis[TD], dataset= dataset, batchsize=3000, num_workers=6)
                del dataset
                label_OHEs.append(label_OHE)
            mean, var = uncertainty.compute_var_mean(label_OHEs, mean, var, ii)
            del label_OHE


        ############################################################################################
        ############################################################################################
        ### compute uncertainty images
        ############################################################################################
        ############################################################################################
        uncertainty.render_uncert_imgs(fns_whole, var, niis,
                                       subjs['train'], numchannels, iterations,species, pkl=pkl, suffix=suffix)
        if validation:
            uncertainty.render_uncert_imgs(valfns_whole, valvar, valniis,
                                           subjs['val'], numchannels, iterations, species, pkl=pkl, suffix='val'+suffix)
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
            uncertniis.append('/data/{5}/variance/{0}_vars_i{2}_{1}ch_en_{3}_{4}.nii.gz'.format(
                subjs['train'][i], numchannels, iterations, i, suffix, species))
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
                    '//data/{5}/variance/{0}_vars_i{2}_{1}ch_en_{3}_val{4}.nii.gz'.format(
                        subjs['val'][i], numchannels, iterations, i,suffix, species))
                # fname = generate_obj_files.generate_mask(valinitinterfeatimgs_for_refstage[i].split('.')[0], valfns_whole[i])
                obj = '//{6}/{4}_{0}ch_valinitinterfeatimg_e{1}_i{3}_{5}'.format(
                    numchannels, epoch,i, smallestidx, subjs['val'][i], suffix,objects_folder)
                valobjs.append('{0}.h5'.format(obj))
                generate_obj_files.generate_h5_files(obj, valinitinterfeatimgs_for_refstage[i],None, vallabels[i])
                uncert = '//{6}/{4}_{0}ch_valuncert_e{1}_f{2}_i{3}_{5}'.format(
                    numchannels, epoch, i,smallestidx,subjs['val'][i], suffix,objects_folder)
                valuncerts.append('{0}.h5'.format(uncert))
                generate_obj_files.generate_h5_files(uncert, valuncertniis[i], None, vallabels[i])


        ############################################################################################
        ############################################################################################
        ### train the 2nd stage model
        ############################################################################################
        ############################################################################################
        # path = '/data/rat/objects/'
        # objs = [path + '{0}_4ch_initinterfeatimg_e10_i4_2022_50slices1.h5'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
        # uncerts = [path + '{0}_4ch_uncert_e10_f{1}_i4_2022_50slices1.h5'.format(subjs['train'][i],i) for i in range(len(subjs['train']))]
        # valobjs = [path + '{0}_4ch_valinitinterfeatimg_e10_i4_2022_50slices1.h5'.format(subjs['val'][i]) for i in range(len(subjs['val']))]
        # valuncerts = [path + '{0}_4ch_valuncert_e10_f{1}_i4_2022_50slices1.h5'.format(subjs['val'][i],i) for i in range(len(subjs['val'])) ]
        # path2 = '/data/rat/intermediate_nii/'
        # path3 = '/data/rat/variance/'
        # initinterfeatimgs_for_refstage = [path2 +'{0}_4ch_initinterfeatimg_e10_i4_2022_50slices1.nii.gz'.format(subjs['train'][i],i) for i in range(len(subjs['train']))]
        # uncertniis = [path3 +'{0}_vars_i5_4ch_en_{1}_2022_50slices1.nii.gz'.format(subjs['train'][i],i) for i in range(len(subjs['train']))]
        # valinitinterfeatimgs_for_refstage = [path2 +'{0}_4ch_valinitinterfeatimg_e10_i4_2022_50slices1.nii.gz'.format(subjs['val'][i],i) for i in range(len(subjs['val']))]
        # valuncertniis = [path3 +'{0}_vars_i5_4ch_en_{1}_val2022_50slices1.nii.gz'.format(subjs['val'][i],i) for i in range(len(subjs['val']))]

        textfn = '/{1}/secondmodel_train_losses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),losses_folder)
        numchannels2 = 10
        epoch = 1
        with open(textfn, 'w') as f:
            f.write("{0}\t{1}\t{2}\n".format('label_losses_cat', 'label_losses_mod_cat', 'total_losses_cat'))
        solver = run_two_stage_cnn_orig_truncatedloss_v22.Solver(objs, epoch=epoch, lr=5e-4, f_dim=numchannels2, batch_size=30000, labels=2,
                                                             shuffle=True, channels=numchannels, pad=pad,  textfn=textfn,
                                                             uncertainty=True, uncertfn= uncerts, channels2=2, valobj=valobjs, valuncertfn=valuncerts,
                                                                 numslices=numslices, slices=slices, axes=axes,
                                                                 dataset_portion=dataset_portion, twoplane=planes
                                                             )
        print('Starting second model training, iteration number {0}'.format(ITER + 1))
        solver.train()
        refmodel = '/{3}/ref_e{0}_f{1}_checkpoint_model_{2}.pth'.format(epoch, numchannels,suffix,checkpoints_folder)
        # solver.model.load_state_dict(torch.load(
        #     '/{3}/ref_e{0}_lr5e4_f{1}_checkpoint_model_{2}.pth'.format(epoch, numchannels,suffix,checkpoints_folder)))
        torch.save(solver.model.state_dict(), refmodel)

        refineinterfeatimgs =[]
        refineoutputs = []
        for v in np.arange(len(objs)):
            refineinterfeatimg = '/{6}/{4}_{0}ch_refineinterfeatimg_added_e{1}_i{3}_{5}.nii.gz'.format(numchannels,
                                                                                                                 epoch, v, 0, subjs['train'][v],suffix,outputs_folder)
            refineoutput = '/{6}/{4}_{0}ch_refineoutput_added_e{1}_f{2}_i{3}_{5}.nii.gz'.format(numchannels, epoch, v, 0,
                                                                                                             subjs['train'][v],suffix,outputs_folder)
            refineinterfeatimgs.append(refineinterfeatimg)
            refineoutputs.append(refineoutput)

        for TD in range(0, len(fns_whole)):
            datastruct = data_preproc_v22.imagepatches(fname=initinterfeatimgs_for_refstage[TD], label=labels[TD], num_classes=2, pad=5,
                                                       masklabel=True, ram=True, normalize=False)
            dataset = datastruct.return_data_struct()
            del datastruct
            datastruct = data_preproc_v22.imagepatches(fname=uncertniis[TD], label=labels[TD],
                                                       num_classes=2, pad=5, normalize=False,
                                                       masklabel=True, ram=True)
            dataset2 = datastruct.return_data_struct()
            del datastruct
            label_OHE = solver.test(refineinterfeatimgs[TD], refineoutputs[TD], niis[TD], dataset=dataset, batchsize=10000,
                                    num_workers=6, dataset2=dataset2, dontWriteInter=True)
            del dataset, dataset2

        # label_OHE = solver.test(refineinterfeatimgs, refineoutputs, niis, batchsize=20000, imgs=objs, uncertfn=uncerts) #,
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

            for TD in range(0, len(valfns_whole)):
                datastruct = data_preproc_v22.imagepatches(fname=valinitinterfeatimgs_for_refstage[TD], label=vallabels[TD],
                                                           num_classes=2, pad=5,
                                                           masklabel=True, ram=True, normalize=False)
                dataset = datastruct.return_data_struct()
                del datastruct
                datastruct = data_preproc_v22.imagepatches(fname=valuncertniis[TD], label=vallabels[TD],
                                                           num_classes=2, pad=5, normalize=False,
                                                           masklabel=True, ram=True)
                dataset2 = datastruct.return_data_struct()
                del datastruct
                label_OHE = solver.test(valrefineinterfeatimgs[TD], valrefineoutputs[TD], valniis[TD], dataset=dataset,
                                        batchsize=10000, num_workers=6, dataset2=dataset2, dontWriteInter=True)
                del dataset, dataset2

            # label_OHE = solver.test(valrefineinterfeatimgs, valrefineoutputs, valniis, batchsize=20000, imgs=valobjs, uncertfn=valuncerts)
            del label_OHE
        ## time
        elapsed = time.time() - starttime
        print(elapsed/60)