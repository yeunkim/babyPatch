import data_preproc_onestage
import data_preproc_noupsample
import matplotlib.pyplot as plt
import nibabel as nib
import pickle

subjs= ['010','023', '039', '056','072', '115','132'] # 087, 002
# t2 = nib.load('//data/infant/T2_train_2021/{0}-C-T1_T2w.1mm.N4.cerebrum.nii.gz'.format('072')).get_fdata()
# plt.hist(t2.ravel()[t2.ravel() > 0], 1000)

image_dir = '/data/infant/T2_train_2021/'
image_file_suffix = '-C-T1_T2w.1mm.bse.N4.full+mask_s3.nii.gz'
mask_dir = '/data/infant/T2_train_2021/'
mask_file_suffix = '-C-T1_T2w.1mm.cerebrum.mask.nii.gz'
label_dir = '/data/infant/t2traindata_labels_handedit_YK_05072020'
label_file_suffix = '-C-T1.T2w.final.label.nii.gz'
save_dir = '/data/infant/objects/'
suffix = "_N4_1mm"

#### whole brain ####
for subj in subjs:
    data0 = data_preproc_noupsample.imagepatches(
        fname='{0}/{1}{2}'.format(image_dir,subj,image_file_suffix),
        mask='{0}/{1}{2}'.format(mask_dir,subj,mask_file_suffix),
        label='{0}/{1}{2}'.format(label_dir,subj,label_file_suffix),
        gm=2, wm=1, csf=3, num_classes=4, pad=5, # k_t2=4, k_t2_init=[300, 60, 640, 950], ## use if normalization goes wrong
        masklabel=True)
    file_obj = open('{0}/{1}{2}'.format(save_dir,subj,suffix), 'wb')
    pickle.dump(data0, file_obj, protocol=4)
    file_obj.close()

#### slices ####
numslices = 10
for subj in subjs:
    pkl_file = '/{0}/{1}{2}.obj'.format(save_dir,subj,suffix)
    file_obj = open(pkl_file, 'rb')
    dataset = pickle.load(file_obj)
    mean = dataset.mean
    std = dataset.std
    del dataset
    data0 = data_preproc_onestage.imagepatches(
        fname='{0}/{1}{2}'.format(image_dir,subj,image_file_suffix),
        mask='{0}/{1}{2}'.format(mask_dir,subj,mask_file_suffix),
        label='{0}/{1}{2}'.format(label_dir,subj,label_file_suffix),
        gm=2, wm=1, csf=3, num_classes=3, pad=5, normfactors= (mean, std), ### always use to have the same mean and standard deviation!!
        masklabel=True, numslicex=numslices, numslicey=numslices, numslicez=numslices)
    file_obj = open('{0}/{1}{2}{3}slices.obj'.format(save_dir,subj,suffix, numslices), 'wb')
    pickle.dump(data0, file_obj, protocol=4)
    file_obj.close()
