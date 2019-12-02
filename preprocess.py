import data_preproc_noupsample
import pickle
import matplotlib.pyplot as plt
import nibabel as nib
import preproc_functions
d=1

subj='1029'
t2 = '/ifs/tmp/mmatern/{0}_train/{0}_T2w.bse.N4.cerebrum.nii.gz'.format(subj)
t2resample = '/ifs/tmp/mmatern/{0}_train/{0}_T2w.bse.N4.cerebrum.1mm.nii.gz'.format(subj)
t2resample_mask = '/ifs/tmp/mmatern/{0}_train/{0}_T2w.bse.N4.cerebrum.1mm.mask.nii.gz'.format(subj)
# t2mask = '/ifs/tmp/mmatern/{0}_train/{0}_T2w.bse.N4.cerebrum.mask.nii.gz'.format(subj)
output = '/data/infant/vae_objs/{0}_T2w_train.obj'.format(subj)

plt.figure()
t2dat = nib.load(t2).get_data()
plt.hist(t2dat.ravel()[t2dat.ravel() > 0], 1000)

preproc_functions.resample1mm(nib.load(t2), t2resample)
preproc_functions.mask(nib.load(t2resample), t2resample_mask)

data0 = data_preproc_noupsample.imagepatches(
        fname=t2resample, mask=t2resample_mask, label= t2resample_mask,
        patchsize=d, upsample=d,  gm=0, wm=1, csf=10, num_classes=5, order=3, pad=5, ysize=5, zsize=1,
        k_t2=4, k_t2_init=[331,100,560,800],channels=1 )

file_obj = open(output, 'wb')
pickle.dump(data0, file_obj, protocol=4)