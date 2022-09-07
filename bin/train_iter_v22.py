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
iterations = 2
classes = 2
skullstrip = True
pad = 5
validation=True
generator = False
epoch = 1
gm2wm = 0.3 # ratio of number of voxels (gm to wm) [if skull stripping -> put brain:non-brain]
csf2wm = 0 # ratio of number of voxels (csf to wm) [0 if skull stripping]
numslices =(10,10,10)
slices = True
axes = (True,True,True)
dataset_portion = 0.1
pkl = False
# planes = (True, False, True)
planes = None
crossfold = 4
second_model = True
numchannels2 = 8
epoch2 = 1
species = 'infant/skull_stripping'
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
# subjs = {'val': ["29312_06222021",
#                  "9261_06252021",
#                 # "d000_inj_pig_f_08.2",
#                 "d030_inj_pig_m_05.3",
#                 # "d030_shm_pig_f_10.2",
#                 # "d100_shm_pig_m_03.4",
#                 "d166_shm_veh_m_11.1"
#                  ],
#          'train': ["29359_06222021",
#                 "29365_06292021",
#                 "9201_07292021",
#                 "9202_07292021",
#                 "d000_inj_veh_m_15.3",
#                 "d000_shm_pig_f_02.1",
#                 "d100_inj_veh_m_04.2",
#                 "d100_shm_veh_f_09.1",
#                    "d166_inj_pig_m_07.1",
#                    "d166_shm_veh_f_09.3"
#                    ]}

subjs = { 'train': ['002', 'V1011','087'],
          'val': []
        }

# maskdir='/home/lucydunnlocal/Final_Train12/Final_Train12/'
# datadir = '/allMouseTrainData/'
datadir = '/run/user/1002/gvfs/sftp:host=miro.dyn.bmap.ucla.edu,user=yeunkimlocal/data/infant/skull_stripping/'
suffix = "_p10"
label = '.T2w.1mm.mask.nii.gz'
anat = '.T2w.1mm.nii.gz'

losses_folder = '/data/{0}/losses/'.format(species)
checkpoints_folder = '/data/{0}/checkpoints/'.format(species)
intermediate_folder = '/data/{0}/intermediate_nii/'.format(species)
outputs_folder = '/data/{0}/outputs/'.format(species)
objects_folder = '/data/{0}/objects/'.format(species)
var_folder = '/data/{0}/variance/'.format(species)
fns_whole = ['/{1}/{0}{2}.h5'.format(subjs['train'][i],objects_folder, suffix) for i in range(len(subjs['train']))]
niis = [nib.load('{0}/{1}{2}'.format(datadir, subjs['train'][i], label))._affine for i in range(len(subjs['train']))]
labels = ['{0}/{1}{2}'.format(datadir, subjs['train'][i], label) for i in range(len(subjs['train']))]
data = ['{0}/{1}{2}'.format(datadir, subjs['train'][i], anat) for i in range(len(subjs['train']))]

suffix = '{0}-{1}-{2}_slices'.format(numslices[0], numslices[1], numslices[2])

print('###############################################################################')
print('###############################################################################')
print('Starting first model training! \n{0}-fold cross validation \nNumber of slices per axis: {1}-{2}-{3} \nAxes to sample from: {4}-{5}-{6}'.format(
    crossfold, numslices[0], numslices[1], numslices[2], axes[0], axes[1],axes[2]))
print('Dataset portion (% of data to be sampled per epoch): {0}'.format(dataset_portion))
print('Optional specification, less planes to be input into model (<3 useful for anisotropic images):', planes)
print('\n')
############################################################################################
############################################################################################
### train the 1st model
############################################################################################
############################################################################################
mean = []
var = []
initinterfeatimgs_alliterations = []
initoutputs_alliterations = []
initimodels = []
starttime = time.time()
for ii in np.arange(iterations):
    label_OHEs = []
    textfn = '/{1}/firstmodel_losses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),losses_folder)
    textfn2 = '/{1}/firstmodel_train_losses_{0}_val.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),
                                                                 losses_folder)
    with open(textfn, 'w') as f:
        f.write("{0}\n".format('label_losses_cat'))
    with open(textfn2, 'w') as f:
        f.write("{0}\n".format('label_losses_cat'))
    print('\nBeginning uncertainty iteration {0}'.format(ii+1))
    solver = run_two_stage_cnn_orig_truncatedloss_v22.Solver(fns_whole, crossfold=crossfold, epoch=epoch, lr=5e-4, f_dim=numchannels, batch_size=10000,
                                    labels=classes, shuffle=True, pad=pad, channels=1, textfn= textfn, suffix=suffix,
                                    numslices= numslices, slices=slices, axes= axes, dataset_portion=dataset_portion, twoplane=planes,
                                    num_workers = 6, dynSize = False #dynSize useful for subcortical area - one stage only
                                    )
    solver.train()
    initmodel = ('/{4}/init_e{1}_{0}ch_i{2}_{3}.pth'.format(numchannels,epoch,ii,suffix,checkpoints_folder))
    torch.save(solver.model.state_dict(), initmodel)
    initimodels.append(initmodel)
    # solver.model.load_state_dict(torch.load(initmodel))
    initinterfeatimgs = []
    initoutputs = []
    for i in np.arange(len(fns_whole)):
        initinterfeatimg = '/{5}/{3}_{0}ch_initinterfeatimg_e{1}_i{2}_{4}.nii.gz'.format(numchannels, epoch, ii, subjs['train'][i], suffix,intermediate_folder)
        initoutput = '/{5}/{3}_{0}ch_initoutput_e{1}_i{2}_{4}.nii.gz'.format(numchannels, epoch, ii, subjs['train'][i],suffix,intermediate_folder)
        initinterfeatimgs.append(initinterfeatimg)
        initoutputs.append(initoutput)
        initinterfeatimgs_alliterations.append(initinterfeatimgs)
        initoutputs_alliterations.append(initoutputs)

    for TD in range(0,len(fns_whole)):
        print("\nGenerating intermediate output for {0}.".format(subjs['train'][TD]))
        datastruct = data_preproc_v22.imagepatches(fname=data[TD], label=labels[TD], num_classes=classes, pad=pad, masklabel=True, ram=True)
        dataset = datastruct.return_data_struct()
        del datastruct
        label_OHE = solver.test(initinterfeatimgs[TD], initoutputs[TD], niis[TD], dataset= dataset, batchsize=5000, num_workers=6)
        del dataset
        label_OHEs.append(label_OHE)
        del label_OHE

    if second_model:
        mean, var = uncertainty.compute_var_mean(label_OHEs, mean, var, ii)
print('\n[*] Training of the first model finished. [*]')

if second_model:
    ############################################################################################
    ############################################################################################
    ### compute uncertainty images
    ############################################################################################
    ############################################################################################
    print('\n Second model initiated. First rendering uncertainty images.')
    uncertainty.render_uncert_imgs(fns_whole, var, niis, subjs['train'], numchannels, iterations,
                                   var_folder, classes=classes, suffix=suffix)
    #### Choose which model/intermediate dataset
    initinterfeatimgs_for_refstage, initoutputs_for_refstage, smallestidx = uncertainty.select_best_model(gm2wm, csf2wm, iterations,
                                                    initoutputs_alliterations, initimodels, initinterfeatimgs_alliterations)

    ############################################################################################
    ############################################################################################
    ### generate obj files
    ############################################################################################
    ############################################################################################
    uncertniis = []
    objs = []
    uncerts = []
    objs_whole = []
    uncerts_whole = []
    print('Generating HDF5 images for the second model.')
    for i in np.arange(len(fns_whole)):
        uncertniis.append('/data/{4}/variance/{0}_vars_i{2}_{1}ch_{3}.nii.gz'.format(
            subjs['train'][i], numchannels, iterations, suffix, species))
        obj = '/{5}/{3}_{0}ch_initinterfeatimg_e{1}_i{2}_{4}'.format(
            numchannels, epoch, smallestidx, subjs['train'][i], suffix,objects_folder)
        objs.append('{0}.h5'.format(obj))
        generate_obj_files.generate_h5_files(obj, initinterfeatimgs_for_refstage[i], None, label = labels[i], pad=pad, skullstrip=skullstrip)
        uncert = '/{5}/{3}_{0}ch_uncert_e{1}_i{2}_{4}'.format(numchannels, epoch, smallestidx,
                                                                    subjs['train'][i], suffix,objects_folder)
        uncerts.append('{0}.h5'.format(uncert))
        generate_obj_files.generate_h5_files(uncert, uncertniis[i], None, label = labels[i], pad=pad, skullstrip=skullstrip)

    ############################################################################################
    ############################################################################################
    ### train the 2nd stage model
    ############################################################################################
    ############################################################################################
    print('\nStarting second model training!')
    # path = '/data/infant/skull_stripping/objects/'
    # objs = [path + '{0}_4ch_initinterfeatimg_e2_i0_2022_10slices1.h5'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
    # uncerts = [path + '{0}_4ch_uncert_e2_f0_i0_2022_10slices1.h5'.format(subjs['train'][i]) for i in range(len(subjs['train']))]
    # valobjs = [path + '{0}_4ch_valinitinterfeatimg_e2_i0_2022_10slices1.h5'.format(subjs['val'][i]) for i in range(len(subjs['val']))]
    # valuncerts = [path + '{0}_4ch_valuncert_e2_f0_i0_2022_10slices1.h5'.format(subjs['val'][i]) for i in range(len(subjs['val'])) ]
    # path2 = '/data/infant/skull_stripping/intermediate_nii/'
    # path3 = '/data/infant/skull_stripping/variance/'
    # initinterfeatimgs_for_refstage = [path2 +'{0}_4ch_initinterfeatimg_e2_i1_2022_10slices1.nii.gz'.format(subjs['train'][i],i) for i in range(len(subjs['train']))]
    # uncertniis = [path3 +'{0}_vars_i2_4ch_en_{1}_2022_10slices1.nii.gz'.format(subjs['train'][i],i) for i in range(len(subjs['train']))]
    # valinitinterfeatimgs_for_refstage = [path2 +'{0}_4ch_valinitinterfeatimg_e2_i1_2022_10slices1.nii.gz'.format(subjs['val'][i],i) for i in range(len(subjs['val']))]
    # valuncertniis = [path3 +'{0}_vars_i2_4ch_en_0_val2022_10slices1.nii.gz'.format(subjs['val'][i]) for i in range(len(subjs['val']))]

    textfn = '/{1}/secondmodel_train_losses_{0}.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'),losses_folder)
    textfn2 = '/{1}/secondmodel_train_losses_{0}_val.txt'.format(datetime.today().strftime('%Y%m%d%h%m%s'), losses_folder)
    with open(textfn, 'w') as f:
        f.write("{0}\n".format('label_losses_cat'))
    with open(textfn2, 'w') as f:
        f.write("{0}\n".format('label_losses_cat'))
    solver = run_two_stage_cnn_orig_truncatedloss_v22.Solver(objs, epoch=epoch2, lr=5e-4, f_dim=numchannels2, batch_size=3000, labels=classes,
                                                         shuffle=True, channels=numchannels, pad=pad,  textfn=textfn,
                                                         uncertainty=True, uncertfn= uncerts, channels2=classes,
                                                             numslices=numslices, slices=slices, axes=axes,
                                                             dataset_portion=dataset_portion, twoplane=planes, num_workers=6
                                                         )
    solver.train()
    refmodel = '/{3}/ref_e{0}_{1}ch_checkpoint_model_{2}.pth'.format(epoch2, numchannels2,suffix,checkpoints_folder)
    # solver.model.load_state_dict(torch.load(refmodel))
    torch.save(solver.model.state_dict(), refmodel)

    refineinterfeatimgs =[]
    refineoutputs = []
    for v in np.arange(len(objs)):
        refineinterfeatimg = '/{4}/{0}_{1}ch_refineinterfeatimg_e{2}_{3}.nii.gz'.format(subjs['train'][v], numchannels2,
                                                                                    epoch2, suffix,outputs_folder)
        refineoutput = '/{4}/{0}_{1}ch_refineoutput_e{2}_{3}.nii.gz'.format(subjs['train'][v], numchannels2,
                                                                                    epoch2, suffix,outputs_folder)

        refineinterfeatimgs.append(refineinterfeatimg)
        refineoutputs.append(refineoutput)

    for TD in range(0, len(fns_whole)):
        datastruct = data_preproc_v22.imagepatches(fname=initinterfeatimgs_for_refstage[TD], label=labels[TD],
                                                   num_classes=classes, pad=pad, masklabel=True, ram=True, normalize=False)
        dataset = datastruct.return_data_struct()
        del datastruct
        datastruct = data_preproc_v22.imagepatches(fname=uncertniis[TD], label=labels[TD], num_classes=classes, pad=pad,
                                                   normalize=False, masklabel=True, ram=True)
        dataset2 = datastruct.return_data_struct()
        del datastruct
        label_OHE = solver.test(refineinterfeatimgs[TD], refineoutputs[TD], niis[TD], dataset=dataset, batchsize=5000,
                                num_workers=6, dataset2=dataset2, dontWriteInter=True)
        del dataset, dataset2
        del label_OHE

print('[*] All training completed! [*]')
elapsed = time.time() - starttime
print('Total time (minutes):', elapsed/60)