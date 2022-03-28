from preprocess import data_preproc_v22

subjs= ['010','023'] # 087, 002
# , '039', '056','072', '115','132'
# t2 = nib.load('//data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.N4.cerebrum.nii.gz'.format('072')).get_fdata()
# plt.hist(t2.ravel()[t2.ravel() > 0], 1000)

image_dir = '/data/infant/T2_train_2021/'
image_file_suffix = '-C-T1_T2w.1mm.bse.N4.full+mask_s3.nii.gz'
mask_dir = '/data/infant/T2_train_2021/'
mask_file_suffix = '-C-T1_T2w.1mm.cerebrum.mask.nii.gz'
label_dir = '/data/infant/t2traindata_labels_handedit_YK_05072020'
label_file_suffix = '-C-T1.T2w.final.label.nii.gz'
save_dir = '/data/infant/objects/'
suffix = "_whole"

#### whole brain ####
for subj in subjs:
    data_preproc_v22.imagepatches(
        fname='{0}/{1}{2}'.format(image_dir,subj,image_file_suffix),
        mask='{0}/{1}{2}'.format(mask_dir,subj,mask_file_suffix),
        label='{0}/{1}{2}'.format(label_dir,subj,label_file_suffix),
        fnoutput='{0}/{1}{2}'.format(save_dir, subj, suffix),
        gm=2, wm=1, csf=3, num_classes=4, pad=5, # k_t2=4, k_t2_init=[300, 60, 640, 950], ## use if normalization goes wrong
        masklabel=True)
