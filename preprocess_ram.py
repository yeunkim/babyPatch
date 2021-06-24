import data_preproc_noupsample
import data_preproc_h5_light
import data_preproc_onestage
import matplotlib.pyplot as plt
import nibabel as nib
import pickle

subjs= ['010','023', '039', '056','072', '115','132'] # 087, 002
# subjs=['072']
# fig, ax = plt.subplots(1,9, sharex=True, sharey=True)
# t2 = nib.load('//data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.N4.cerebrum.nii.gz'.format('072')).get_fdata()
# plt.hist(t2.ravel()[t2.ravel() > 0], 1000)
# for i, subj in enumerate(subjs):
#     t2 = nib.load('//data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.N4.cerebrum.nii.gz'.format(subj)).get_fdata()
#     ax[i].hist(t2.ravel()[t2.ravel() > 0], 1000)
    # plt.hist(t2.ravel()[t2.ravel() > 0], 1000)

# for subj in subjs:
#     data0 = data_preproc_noupsample.imagepatches(
#         fname='/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.bse.N4.full+mask_s3.nii.gz'.format(subj),
#         mask='//data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subj),
#         label='//data/infant/t2traindata_labels_handedit_YK_05072020/{0}-C-T1.T2w.final.label.nii.gz'.format(subj),
#         gm=2, wm=1, csf=3, num_classes=4, pad=5,k_t2=4, k_t2_init=[300, 60, 640, 950],
#         masklabel=True) #,  k_t2=4, k_t2_init=[290, 100, 760, 940])
#     file_obj = open('/data/infant/objects/{0}_N4_1mm.obj'.format(subj), 'wb')
#     pickle.dump(data0, file_obj, protocol=4)
#     file_obj.close()

for subj in subjs:
    pkl_file = '/data/infant/objects/{0}_p5_20slices.obj'.format(subj)
    file_obj = open(pkl_file, 'rb')
    dataset = pickle.load(file_obj)
    mean = dataset.mean
    std = dataset.std
    del dataset
    data0 = data_preproc_onestage.imagepatches(
        fname='/data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.bse.N4.full+mask_s3.nii.gz'.format(subj),
        mask='//data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.cerebrum.mask.nii.gz'.format(subj),
        label='//data/infant/t2traindata_labels_handedit_YK_05072020/{0}-C-T1.T2w.final.label.nii.gz'.format(subj),
        gm=2, wm=1, csf=3, num_classes=3, pad=5, normfactors= (mean, std), #k_t2=4, k_t2_init=[300, 60, 640, 950],
        masklabel=True, numslicex=10, numslicey=10, numslicez=10)
    file_obj = open('/data/infant/objects/{0}_p5_10slices.obj'.format(subj), 'wb')
    pickle.dump(data0, file_obj, protocol=4)
    file_obj.close()

# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data_mult/002/002-C-T1_T2w.1mm.N4.cerebrum.nii.gz',
#     mask='/mnt/data/infant/processed/train_data_mult/002/002-T2w_resampled.cerebrum.mask.nii.gz',
#     label='/mnt/data/infant/processed/train_data/002/002.label4.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=5, k_t2_init=[100, 30, 230, 150, 20])
# file_obj = open('/mnt/data/infant/h5data/train_raw/002_T2_1mm_test.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()
#
# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data_mult/002/002-C-T1_t1w.N4.1mm.cerebrum.nii.gz' ,
#     mask='/mnt/data/infant/processed/train_data_mult/002/002-T2w_resampled.cerebrum.mask.nii.gz',
#     label='/mnt/data/infant/processed/train_data/002/002.label4.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=5, k_t2_init=[310, 130, 600, 20, 2500])
# file_obj = open('/mnt/data/infant/h5data/train_raw/002_T1_1mm_test.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data/002/002-T2w_resampled.N4.cerebrum.handcraft3.nii.gz',
#     mask='/mnt/data/infant/processed/train_data/002/002.handcraft.smooth.nii.gz',
#     label='/mnt/data/infant/processed/train_data/002/002.label4.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=5, k_t2_init=[330, 100, 800, 20, 600])
# file_obj = open('/mnt/data/infant/h5data/train_raw/002_handcraft_1mm.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data/002/002-T2w_resampled.N4.cerebrum.erode.nii.gz',
#     mask='/mnt/data/infant/processed/train_data/002/002-T2w_resampled.cerebrum.mask.erode.nii.gz',
#     label='/mnt/data/infant/processed/train_data/002/002.label2.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=4, k_t2_init=[330, 100, 800, 20])
# file_obj = open('/mnt/data/infant/h5data/train_raw/002_erode_edit11_1mm.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data/002/002-T2w_resampled.cerebrum.nii.gz',
#     mask='/mnt/data/infant/processed/train_data/002/002-T2w_resampled.cerebrum.subcort.mask.nii.gz',
#     label='/mnt/data/infant/processed/train_data/002/002.label4.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=4, k_t2_init=[330, 100, 800, 20])
# file_obj = open('/mnt/data/infant/h5data/train_raw/002_nobias_1mm.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

## rotations
# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data/002/002-T2w_resampled.N4.cerebrum_rot15.ax12.nii.gz',
#     mask='/mnt/data/infant/processed/train_data/002/002-T2w_resampled.N4.cerebrum_rot15.ax12.nii.gz',
#     label='/mnt/data/infant/processed/train_data/002/002.label3.rot15.ax12.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=5, k_t2_init=[330, 100, 800, 20, 600])
# file_obj = open('/mnt/data/infant/h5data/train_raw/002_bias_1mm_rot15_ax12.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()
#
# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data/002/002-T2w_resampled.N4.cerebrum_rot-15.ax12.nii.gz',
#     mask='/mnt/data/infant/processed/train_data/002/002-T2w_resampled.N4.cerebrum_rot-15.ax12.nii.gz',
#     label='/mnt/data/infant/processed/train_data/002/002.label3.rot-15.ax12.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=5, k_t2_init=[330, 100, 800, 20, 600])
# file_obj = open('/mnt/data/infant/h5data/train_raw/002_bias_1mm_rot-15_ax12.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

## 087

data0 = data_preproc_noupsample.imagepatches(
    fname='/mnt/data/infant/processed/train_data/087/087/087-C-T1_T2w.1mm.N4.cerebrum.nii.gz',
    mask='/mnt/data/infant/processed/train_data/087/087/087-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz',
    label='/mnt/data/infant/processed/train_data/087/087.label.whole5.nii.gz',
    gm=2, wm=1, csf=3, num_classes=4, pad=5,
    masklabel=True, k_t2=4, k_t2_init=[100, 27, 170, 300])
file_obj = open('/data/infant/objects/087_p6_2.obj', 'wb')
pickle.dump(data0, file_obj, protocol=4)
file_obj.close()


# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data/087/087/087-C-T1_T2w.1mm.cerebrum.nii.gz',
#     mask='/mnt/data/infant/processed/train_data/087/087/087-C-T1_T2w.1mm.cerebrum.subcort.mask.nii.gz',
#     label='/mnt/data/infant/processed/train_data/087/087.label.whole3.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=4, k_t2_init=[100, 27, 170, 300])
# file_obj = open('/mnt/data/infant/h5data/train_raw/087_nobias_1mm.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

# rotations

# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data/087/087-C-T1_T2w.1mm.N4.cerebrum_rot15.ax12.nii.gz',
#     mask='/mnt/data/infant/processed/train_data/087/087-C-T1_T2w.1mm.N4.cerebrum_rot15.ax12.nii.gz',
#     label='/mnt/data/infant/processed/train_data/087/087.label.whole3.rot15.ax12.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=4, k_t2_init=[100, 27, 170, 300])
# file_obj = open('/mnt/data/infant/h5data/train_raw/087_bias_1mm_rot15_ax12.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()
#
# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data/087/087-C-T1_T2w.1mm.N4.cerebrum_rot-15.ax12.nii.gz',
#     mask='/mnt/data/infant/processed/train_data/087/087-C-T1_T2w.1mm.N4.cerebrum_rot-15.ax12.nii.gz',
#     label='/mnt/data/infant/processed/train_data/087/087.label.whole3.rot-15.ax12.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=4, k_t2_init=[100, 27, 170, 300])
# file_obj = open('/mnt/data/infant/h5data/train_raw/087_bias_1mm_rot-15_ax12.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

# subj='010'
subj='108-C-T1'
# data0 = data_preproc_noupsample.imagepatches(
#     fname='/oldmiro/data/SSD_data/infant/processed/test_data/010/010-C-T1_T2w.1mm.N4.cerebrum.nii.gz',
#     mask='/oldmiro/data/SSD_data/infant/processed/test_data/010/010-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz',
#     label='/oldmiro/data/SSD_data/infant/processed/test_data/010/010.label4.nii.gz',
#     spherecoords= '/data/infant/spherecoords/atlas_Int_M6_T2_cartesian_new_coords_masked_{0}_xfmed_spherecoord_filled.nii.gz'.format(subj),
#     gm=2, wm=1, csf=3, num_classes=4, pad=6,
#     masklabel=True, k_t2=4, k_t2_init=[110, 30, 200, 300])
# file_obj = open('/data/infant/objects/010_p6_spherecoords.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

data0 = data_preproc_noupsample.imagepatches(
    fname='/nafs/shattuck/yeunkim/infant_images/rebeccabelisle/{0}_T2w.1mm.N4.cerebrum.nii.gz'.format(subj),
    mask='/nafs/shattuck/yeunkim/infant_images/rebeccabelisle/{0}_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj),
    # mask = '//oldmiro/data/SSD_data/infant/processed/train_data/{0}/{0}_T2w.1mm.N4.subcort.mask.nii.gz'.format('072'),
    # label='/oldmiro/data/SSD_data/infant/processed/train_data/087/087.label.whole4.nii.gz',
    label = '/oldmiro/data/SSD_data/infant/processed/train_data/072/072.label.nii.gz',
    spherecoords= '/data/infant/spherecoords/atlas_Int_M6_T2_cartesian_new_coords_masked_{0}_xfmed_spherecoord2_filled.nii.gz'.format(subj),
    gm=2, wm=1, csf=3, num_classes=4, pad=6,
    masklabel=True)
file_obj = open('/data/infant/objects/{0}_p6_spherecoords.obj'.format(subj), 'wb')
pickle.dump(data0, file_obj, protocol=4)
file_obj.close()

data0 = data_preproc_noupsample.imagepatches(
    fname='/mnt/data/infant/processed/test_data/010/010-C-T1_T2w.1mm.N4.cerebrum.nii.gz',
    mask='/ifshome/yeunkim/infant_labels/010/010_T2w.1mm.N4.cerebrum.mask.nii.gz',
    label='/mnt/data/infant/processed/test_data/010/010.label4.nii.gz',
    gm=2, wm=1, csf=3, num_classes=4, pad=5,
    masklabel= True, k_t2=4, k_t2_init=[110, 30, 200, 300])
file_obj = open('/mnt/data/infant/h5data/train_raw/010_1mm.obj', 'wb')
pickle.dump(data0, file_obj, protocol=4)
file_obj.close()

data0 = data_preproc_noupsample.imagepatches(
    fname='/mnt/data/infant/processed/train_data_mult/010/010-C-T1_T2w.1mm.N4.cerebrum.nii.gz',
    mask='/mnt/data/infant/processed/train_data_mult/010/010-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz',
    label='/mnt/data/infant/processed/test_data/010/010.label5.nii.gz',
    gm=2, wm=1, csf=3, num_classes=4, pad=5,
    masklabel=True, k_t2=5, k_t2_init=[270, 136, 1400, 20, 1200])
file_obj = open('/mnt/data/infant/h5data/train_raw/010_T2_1mm_test.obj', 'wb')
pickle.dump(data0, file_obj, protocol=4)
file_obj.close()

data0 = data_preproc_noupsample.imagepatches(
    fname='/mnt/data/infant/processed/train_data_mult/010/010-C-T1_t1w.N4.1mm.cerebrum.nii.gz',
    mask='/mnt/data/infant/processed/train_data_mult/010/010-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz',
    label='/mnt/data/infant/processed/test_data/010/010.label5.nii.gz',
    gm=2, wm=1, csf=3, num_classes=4, pad=5,
    masklabel=True, k_t2=5, k_t2_init=[110, 35, 800, 280, 5])
file_obj = open('/mnt/data/infant/h5data/train_raw/010_T1_1mm_test.obj', 'wb')
pickle.dump(data0, file_obj, protocol=4)
file_obj.close()

############ 025

data0 = data_preproc_noupsample.imagepatches(
    fname='/mnt/data/infant/processed/train_data_mult/025/025-C-T1_T2w.1mm.N4.cerebrum.nii.gz',
    mask='/mnt/data/infant/processed/train_data_mult/025/025-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz',
    label='/mnt/data/infant/processed/train_data_mult/025/025-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz',
    gm=2, wm=1, csf=3, num_classes=4, pad=5,
    masklabel=True, k_t2=5, k_t2_init=[85, 17, 270, 200, 150])
file_obj = open('/mnt/data/infant/h5data/train_raw/025_T2_1mm_test.obj', 'wb')
pickle.dump(data0, file_obj, protocol=4)
file_obj.close()

data0 = data_preproc_noupsample.imagepatches(
    fname='/mnt/data/infant/processed/train_data_mult/025/025-C-T1_t1w.N4.1mm.cerebrum.nii.gz',
    mask='/mnt/data/infant/processed/train_data_mult/025/025-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz',
    label='/mnt/data/infant/processed/train_data_mult/025/025-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz',
    gm=2, wm=1, csf=3, num_classes=4, pad=5,
    masklabel=True, k_t2=5, k_t2_init=[260, 150, 500, 20, 2500])
file_obj = open('/mnt/data/infant/h5data/train_raw/025_T1_1mm_test.obj', 'wb')
pickle.dump(data0, file_obj, protocol=4)
file_obj.close()

# subj = '025'
# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.N4.cerebrum.nii.gz'.format(subj),
#     mask='/mnt/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj),
#     label='/mnt/data/infant_2019/transformed_labels/{0}/{0}/{0}-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj),
#     gm=0, wm=1, csf=10, num_classes=4, pad=5, k_t2=4, k_t2_init=[90, 20, 200, 300]
# )
# file_obj = open('/mnt/data/infant/h5data/train_raw/{0}_1mm.obj'.format(subj), 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()
#
# subj = '025'
# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/{0}-skullstripped_anat.nii'.format(subj),
#     mask='/mnt/data/{0}-skullstripped_anat.nii'.format(subj),
#     label='/mnt/data/{0}-skullstripped_anat.mask.nii'.format(subj),
#     gm=0, wm=1, csf=10, num_classes=4, pad=5, k_t2=4, k_t2_init=[90, 20, 200, 300]
# )
# file_obj = open('/mnt/data/infant/h5data/train_raw/{0}_1mm_ibeat.obj'.format(subj), 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

# subj = '010'
# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/test_data/{0}/{0}-C-T1_T2w.1mm.N4.cerebrum.nii.gz'.format(subj),
#     mask='/mnt/data/infant/processed/test_data/{0}/{0}-C-T1_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj),
#     label='/ifshome/yeunkim/infant_labels/in_progress/010_AG.label.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5, k_t2=4, k_t2_init=[110, 30, 200, 330],
#     masklabel=True
# )
# file_obj = open('/mnt/data/infant/h5data/train_raw/{0}_1mm.obj'.format(subj), 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()



subj = '108'

# t2 = nib.load('/mnt/data/infant/processed/train_data/{0}/{0}_T2w.1mm.N4.cerebrum.nii.gz'.format(subj)).get_fdata()
# plt.hist(t2.ravel()[t2.ravel() > 0], 1000)

data0 = data_preproc_noupsample.imagepatches(
    fname='/mnt/data/infant/processed/train_data/{0}/{0}_T2w.1mm.N4.cerebrum.nii.gz'.format(subj),
    mask='/mnt/data/infant/processed/train_data/{0}/{0}_T2w.1mm.N4.subcort.mask.nii.gz'.format(subj),
    label='/mnt/data/infant/processed/train_data/{0}/{0}.label.nii.gz'.format(subj),
    gm=2, wm=1, csf=3, num_classes=4, pad=5,
    masklabel=True, k_t2=4, k_t2_init=[100, 30, 200, 300])
file_obj = open('/mnt/data/infant/h5data/train_raw/{0}_1mm.obj'.format(subj), 'wb')
pickle.dump(data0, file_obj, protocol=4)
file_obj.close()

# data0 = data_preproc_noupsample.imagepatches(
#     fname='/mnt/data/infant/processed/train_data/115/115_T2w.1mm.N43.bse.nii.gz',
#     mask='/mnt/data/infant/processed/train_data/115/115_T2w.1mm.N43.subcort.mask.nii.gz',
#     label='/mnt/data/infant/processed/train_data/115/115.label.nii.gz',
#     gm=2, wm=1, csf=3, num_classes=4, pad=5,
#     masklabel=True, k_t2=4, k_t2_init=[33, 10, 80, 60])
# file_obj = open('/mnt/data/infant/h5data/train_raw/115_subcort_1mm.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)
# file_obj.close()

subjs = ['021', '027','035', '036', '039','040', '045']
subjs = ['020']
# t2 = nib.load('/data/infant/processed/test_data/{0}/{0}_T2w.1mm.N4.cerebrum.nii.gz'.format(subjs[7])).get_fdata()
# plt.hist(t2.ravel()[t2.ravel() > 0], 1000)
for subj in subjs:
    data0 = data_preproc_noupsample.imagepatches(
        fname='/data/infant/processed/test_data/{0}/{0}_T2w.1mm.N4.cerebrum.nii.gz'.format(subj),
        mask='/data/infant/processed/test_data/{0}/{0}_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj),
        label='/data/infant/processed/test_data/{0}/{0}_T2w.1mm.N4.cerebrum.mask.nii.gz'.format(subj),
        gm=2, wm=1, csf=3, num_classes=4, pad=5,
        masklabel=False, k_t2=4, k_t2_init=[100, 30, 200, 300])
    file_obj = open('/mnt/data/infant/h5data/train_raw/{0}_1mm.obj'.format(subj), 'wb')
    pickle.dump(data0, file_obj, protocol=4)
    file_obj.close()

subjs = ['025', '027', '031', '036', '039', '040']
t2 = nib.load('/kahlo/data/T1{0}-5/{0}-skullstripped_anat.nii'.format(subjs[5])).get_fdata()
plt.hist(t2.ravel()[t2.ravel() > 0], 1000)
subj = subjs[5]
data0 = data_preproc_noupsample.imagepatches(
    fname='/kahlo/data/T1{0}-5/{0}-skullstripped_anat.nii'.format(subj),
    mask='/kahlo/data/T1{0}-5/{0}-skullstripped_anat.nii'.format(subj),
    label='/kahlo/data/T1{0}-5/{0}.TD2.label.nii.gz'.format(subj),
    gm=2, wm=1, csf=3, num_classes=4, pad=5,
    masklabel=False, k_t2=5, k_t2_init=[500, 121, 1018, 1427, 38])
file_obj = open('/mnt/data/infant/h5data/train_raw/{0}_ibeatspace_1mm.obj'.format(subj), 'wb')
pickle.dump(data0, file_obj, protocol=4)
file_obj.close()