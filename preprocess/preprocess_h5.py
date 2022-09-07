from preprocess import data_preproc_v22

# subjs= ['sham', 'injured24h'] # 087, 002
# # , '039', '056','072', '115','132'
# # t2 = nib.load('//data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.N4.cerebrum.nii.gz'.format('072')).get_fdata()
# # plt.hist(t2.ravel()[t2.ravel() > 0], 1000)
#
# image_dir = '/data/mouse/Neil_TBI/T2highres_'
# image_file_suffix = '.nii.gz'
# mask_dir = '/data/mouse/Neil_TBI/T2highres_'
# mask_file_suffix = '.mask.nii.gz'
# label_dir = '/data/infant/t2traindata_labels_handedit_YK_05072020'
# label_file_suffix = '-C-T1.T2w.final.label.nii.gz'
# save_dir = '/data/mouse/objects/'
# suffix = "_whole"
#
# #### whole brain ####data
# for subj in subjs:
#     data_preproc_v22.imagepatches(
#         fname='{0}{1}{2}'.format(image_dir,subj,image_file_suffix),
#         label='{0}{1}{2}'.format(image_dir,subj,mask_file_suffix),
#         fnoutput='{0}/{1}{2}'.format(save_dir, subj, suffix),
#         num_classes=2, pad=5, # k_t2=4, k_t2_init=[300, 60, 640, 950], ## use if normalization goes wrong
#         masklabel=True)

# subjs = []
# with open('/data/rat/Neil_TBI/trainIDs_half.txt', 'r') as f:
#     for id in f.readlines():
#         # print(id.rstrip())
#         subjs.append(id.rstrip())

subjs = ['inj_023_12d','shm_027_12d']

# subjs = [
#         "29312_06222021",
#         "29322_07132021",
#         "29332_07142021",
#         "29358_06222021",
#         "29359_06222021",
#         "29365_06292021",
#         "9201_07292021",
#         "9202_07292021",
#         "9243_07142021",
#         "9244_07152021",
#         "9254_06282021",
#         "9261_06252021",
#         "d000_inj_pig_f_08.2",
#         "d000_inj_veh_m_15.3",
#         "d000_shm_pig_f_02.1",
#         "d000_shm_pig_m_12.2",
#         "d030_inj_veh_f_06.2",
#         "d030_inj_pig_m_05.3",
#         "d030_shm_pig_f_10.2",
#         "d030_shm_pig_m_12.3",
#         "d100_inj_pig_f_14.3",
#         "d100_inj_veh_m_04.2",
#         "d100_shm_veh_f_09.1",
#         "d100_shm_pig_m_03.4",
#         "d166_inj_pig_f_06.3",
#         "d166_inj_pig_m_07.1",
#         "d166_shm_veh_f_09.3",
#         "d166_shm_veh_m_11.1"
#          ]

image = '/2T2star_MEAN.nii.gz'
# label = '/bet_mask.reorient.nii.gz'
# label = '/brain.reorient.resampled0.1mm.int16.label.nii.gz'
label = '/ants_initial_BrainExtractionMask.yk.nii.gz'
maskdir = '/data/rat/Neil_TBI/Train12/'
# maskdir =  '/data/mouse/allMouseTrainData/'
datadir = '/home/lucydunnlocal/Final_Train12/Final_Train12/'
save_dir = '/data/rat/objects/'
suffix = "_p10_xz"

subjs = []
with open('/run/user/1002/gvfs/sftp:host=miro.dyn.bmap.ucla.edu,user=yeunkimlocal'
          '/data/infant/T2_train_2021/MDT_ANTS/trainData/subjs.txt', 'r') as f:
    lines = f.readlines()
    for id in lines:
        subjs.append(id.rstrip().lstrip())
## for infant skull-stripping
datadir = '/run/user/1002/gvfs/sftp:host=miro.dyn.bmap.ucla.edu,user=yeunkimlocal/data/infant/skull_stripping/'
label = '.T2w.1mm.mask.nii.gz'
maskdir = datadir
image = '.T2w.1mm.nii.gz'
save_dir = '/data/infant/skull_stripping/objects/'
suffix = "_p10"

for subj in subjs:
    data_preproc_v22.imagepatches(
        fname='{0}{1}{2}'.format(datadir,subj,image),
        label='{0}{1}{2}'.format(maskdir,subj,label),
        fnoutput='{0}/{1}{2}'.format(save_dir, subj, suffix),
        num_classes=2, pad=10, # k_t2=4, k_t2_init=[300, 60, 640, 950], ## use if normalization goes wrong
        masklabel=False)

# data_preproc_v22.imagepatches(
#     fname='/home/yeunkimlocal/Downloads/shm_015_12d.rat.test.mask.nii.gz',
#     fnoutput='/data/rat/objects/shm_015_12d_whole.test',
#     num_classes=2, pad=5)
