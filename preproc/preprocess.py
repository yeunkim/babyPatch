import data_preproc_noupsample
import pickle
from preproc_functions import resample1mm
import matplotlib.pyplot as plt
import nibabel as nib
import preproc_functions
import numpy as np
d=1

subj='040'
t2 = '/ifs/tmp/mmatern/{0}_train_re/{0}_re_T2w.bse.N4.cerebrum.nii.gz'.format(subj)
t2 = '/data/infant/processed/test/{0}_vars_10.nii.gz'.format(subj)
# t2 = '/data/Mouse_Test/MKO4-BN/121_BC_masked.nii'
t2resample = '/ifs/tmp/mmatern/{0}_train_re/{0}_re_T2w.bse.N4.cerebrum.nii.gz'.format(subj)
t2resample_mask = '/ifs/tmp/mmatern/{0}_train_re/{0}_re_T2w.bse.N4.cerebrum.mask.nii.gz'.format(subj)
t2resample_mask = '/data/T1002/T1002-5/002-skullstripped_anat.cerebrum.mask.nii'
t2resample_mask = '/data/T1{0}-5/{0}-skullstripped_anat.nii'.format(subj)
# t2resample_mask = '/data/Mouse_Test/MKO4-BN121_BC_masked.cerebrum.mask.nii'
# label = '/data/Mouse_Test/MKO4-BN121_BC.label.nii.gz'
# t2mask = '/ifs/tmp/mmatern/{0}_train/{0}_T2w.bse.N4.cerebrum.mask.nii.gz'.format(subj)
output = '/data/infant/processed/train_data/{0}/{0}_T2w_train_morecsf.obj'.format(subj)
output = '/data/infant/vae_objs/{0}_vars_10.obj'.format(subj)
# output = '/data/mouse.obj'

# plt.figure()
# t2dat = nib.load(t2).get_data()
# plt.hist(t2dat.ravel()[t2dat.ravel() > 0], 1000)
#
# preproc_functions.resample1mm(nib.load(t2), t2resample)
# preproc_functions.mask(nib.load(t2resample), t2resample_mask)
#
# data0 = data_preproc_noupsample.imagepatches(
#         fname=t2resample, mask=t2resample_mask, label= t2resample_mask,
#         patchsize=d, upsample=d,  gm=2, wm=1, csf=3, num_classes=5, order=3, pad=5, ysize=5, zsize=1,
#         k_t2=4, k_t2_init=[13000,31000,3600,25000])

data0 = data_preproc_noupsample.imagepatches(
        fname=t2, mask=t2resample_mask, label= t2resample_mask,
        patchsize=d, upsample=d,  gm=2, wm=1, csf=3, num_classes=5, order=3, pad=5, ysize=5, zsize=1, channels=3)

file_obj = open(output, 'wb')
pickle.dump(data0, file_obj, protocol=4)

#####################
subj='002'

degrees = [20, 10, -10, -20, -10, 10]
axes = [12, 12, 12, 20, 20, 20]

degrees = [10, -10, -10, 10]
axes = [12, 12, 20, 20]

t2ref = '/data/infant/processed/train_data/{0}/002-T2w_resampled.bse.N4.nii.gz'.format('002')
niiref = nib.load(t2ref)

subjs = ['087', '112', '1029', '002']
subjs = ['1036']
# t2 = '/data/infant/processed/train_data/{0}-T2w_resampled.cerebrum.N4.nii.gz'.format(subj)
# t2= '/data/infant/processed/train_data/{0}-C-T1_T2w.nii.gz'.format(subj)
# t2= '/ifs/tmp/mmatern/{0}_train_2/{0}_train_2_T2w.bse.N4.cerebrum.nii.gz'.format('112')
for subj in subjs:
    t2= '/ifshome/yeunkim/infant_labels/train/{0}-C-T1_T2w.nii.gz'.format(subj)
    t2 = '/data/infant_2019/T2w/{0}-L-T3_T2w.nii'.format(subj)
    # t2= '/ifs/tmp/mmatern/{0}_train_2/{0}_T2w.bse.N4.cerebrum.nii.gz'.format('087')
    # t2out = '/data/infant/processed/train_data/{0}-C-T1_T2w_1mm_order3nearest.nii.gz'.format(subj)
    t2out = '/data/infant/processed/train_data/{0}/{0}-C-T1_T2w.1mm.nii.gz'.format(subj)
    # t2out = '/data/infant/processed/087_T2w.bse.N4.cerebrum.1mm.nii.gz'
    mask = '/data/infant/processed/train_data/002-T2w_resampled.cerebrum.mask2.nii.gz'
    label= '/data/infant/processed/train_data/002-T2.MRM.YK.label.nii.gz'
    coordfn = '/data/infant/processed/train_data/002-T2.coord.nii.gz'

    tmp = nib.load(t2)
    resample1mm(tmp, t2out, order=3, mode='nearest', ref=niiref)

subjs = ['010']

for subj in subjs:
    t2= '/data/infant_2019/T2w/{0}-C-T1_T2w.nii.gz'.format(subj)
    t2out = '/data/infant/processed/test_data/{0}/{0}-C-T1_T2w.1mm.nii.gz'.format(subj)

    tmp = nib.load(t2)
    resample1mm(tmp, t2out, order=3, mode='nearest', ref=niiref)


# orig = data_preproc_noupsample.imagepatches(
#         fname=t2, mask=mask, label= label,
#         patchsize=d, upsample=d,  gm=2, wm=1, csf=3, num_classes=5, order=3, pad=5, ysize=5, zsize=1,
#         k_t2=4, k_t2_init=[330,100,800,40])

data0 = data_preproc_noupsample.imagepatches(
        fname=t2, mask=mask, label= label,
        patchsize=d, upsample=d,  gm=2, wm=1, csf=3, num_classes=5, order=3, pad=5, ysize=5, zsize=1,
        k_t2=4, k_t2_init=[330,100,800,40])
file_obj = open('/data/infant/vae_objs/002-T2w_resampled.cerebrum.N4.rot.orig.obj'.format(0), 'wb')
pickle.dump(data0, file_obj, protocol=4)

#
# tmpdata = data_preproc_noupsample.imagepatches(
#         fname=t2, mask=mask, label= label,
#         patchsize=d, upsample=d,  gm=2, wm=1, csf=3, num_classes=5, order=3, pad=5, ysize=5, zsize=1,
#         k_t2=4, k_t2_init=[330,100,800,40], coords=coordfn)
# # #
t2 = '/data/infant/processed/train_data/{0}-T2w_resampled.cerebrum.nii.gz'.format(subj)
mask = '/data/infant/processed/train_data/002-T2w_resampled.cerebrum.mask2.subcort.nii.gz'
label= '/data/infant/processed/train_data/002-T2.MRM.YK.label.nii.gz'

nobfc = data_preproc_noupsample.imagepatches(
        fname=t2, mask=mask, label= label,
        patchsize=d, upsample=d,  gm=2, wm=1, csf=3, num_classes=5, order=3, pad=5, ysize=5, zsize=1,
        k_t2=4, k_t2_init=[330,100,800,40])

t2 = '/data/infant/processed/train_data/{0}-T2w_resampled.bse.N4.nii.gz'.format(subj)
mask = '/data/infant/processed/train_data/002-T2w_resampled.cerebrum.mask.erode.nii.gz'
label= '/data/infant/processed/train_data/002-T2.MRM.YK.label.nii.gz'

nobse = data_preproc_noupsample.imagepatches(
        fname=t2, mask=mask, label= label,
        patchsize=d, upsample=d,  gm=2, wm=1, csf=3, num_classes=5, order=3, pad=5, ysize=5, zsize=1,
        k_t2=4, k_t2_init=[330,100,800,40])

data0.indices = np.concatenate([data0.indices, nobfc.indices, nobse.indices])
data0.neighbors = np.concatenate([data0.neighbors, nobfc.neighbors, nobfc.neighbors])
data0.neighbors_y = np.concatenate([data0.neighbors_y, nobfc.neighbors_y, nobfc.neighbors_y])
data0.neighbors_z = np.concatenate([data0.neighbors_z, nobfc.neighbors_z, nobfc.neighbors_z])
data0.X = np.concatenate([data0.X, nobfc.X, nobfc.X])
data0.X5 = np.concatenate([data0.X5, nobfc.X5, nobfc.X5])
#data0.coordsvec = np.concatenate([orig.coordsvec, nobfc.coordsvec, nobfc.coordsvec])

file_obj = open('/data/infant/processed/train_data/002-T2w_resampled.cerebrum.N4.augerode.obj'.format(0), 'wb')
pickle.dump(data0, file_obj, protocol=4)

datalist = []

for i in range(0, len(degrees)):
    t2rot = '/data/infant/processed/train_data/{0}-T2w_resampled.cerebrum.N4_rot{1}.ax{2}.nii.gz'.format(subj, degrees[i], axes[i])
    labelrot = '/data/infant/processed/train_data/{0}-T2.MRM.YK.label.masked.rot{1}.ax{2}.nii.gz'.format(subj, degrees[i], axes[i])
    coordrot = '/data/infant/processed/train_data/{0}-T2.coord.rot{1}.ax{2}.nii.gz'.format(subj, degrees[i],
                                                                                                 axes[i])

    tmp = data_preproc_noupsample.imagepatches(
            fname=t2rot, mask=t2rot, label=label,
            patchsize=d, upsample=d, gm=2, wm=1, csf=3, num_classes=5, order=3, pad=5, ysize=5, zsize=1,
            k_t2=4, k_t2_init=[330, 100, 800, 40], masklabel=True)

    print('Finished {0}'.format(i))

    # datalist.append(tmp)

    data0.indices = np.concatenate([data0.indices, tmp.indices + len(data0.indices)])
    data0.neighbors = np.concatenate([data0.neighbors, tmp.neighbors])
    data0.neighbors_y = np.concatenate([data0.neighbors_y, tmp.neighbors_y])
    data0.neighbors_z = np.concatenate([data0.neighbors_z, tmp.neighbors_z])
    data0.X = np.concatenate([data0.X, tmp.X])
    data0.X5 = np.concatenate([data0.X5, tmp.X5])
    #data0.coordsvec = np.concatenate([data0.coordsvec, tmp.coordsvec])

    del tmp

file_obj = open('/data/infant/processed/train_data/002-T2w_resampled.cerebrum.N4.rot.4.test.obj', 'wb')
pickle.dump(data0, file_obj, protocol=4)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# from operator import itemgetter

np.random.seed(0)
# np.random.shuffle(orig.coordsvec)
idxs = np.arange(len(orig.coordsvec))
np.random.shuffle(idxs)
subsets = list(chunks(idxs, 1000))
# allcoords = data0.coordsvec


for i in range(0,len(subsets)):
    tmpdata = data_preproc_noupsample.imagepatches(
        fname=t2, mask=mask, label=label,
        patchsize=d, upsample=d, gm=2, wm=1, csf=3, num_classes=5, order=3, pad=5, ysize=5, zsize=1,
        k_t2=4, k_t2_init=[330, 100, 800, 40], coords=coordfn)

    tmpdata.indices = tmpdata.indices[subsets[i]]
    tmpdata.neighbors = tmpdata.neighbors[subsets[i]]
    tmpdata.neighbors_y = tmpdata.neighbors_y[subsets[i]]
    tmpdata.neighbors_z = tmpdata.neighbors_z[subsets[i]]
    tmpdata.X = tmpdata.X[subsets[i]]
    tmpdata.X5 = tmpdata.X5[subsets[i]]
    tmpdata.coordsvec = tmpdata.coordsvec[subsets[i]]

    for i2 in range(0, len(datalist)):
        tmpdata.indices = np.concatenate([tmpdata.indices, datalist[i2].indices[subsets[i]]])
        tmpdata.neighbors = np.concatenate([tmpdata.neighbors, datalist[i2].neighbors[subsets[i]]])
        tmpdata.neighbors_y = np.concatenate([tmpdata.neighbors_y, datalist[i2].neighbors_y[subsets[i]]])
        tmpdata.neighbors_z = np.concatenate([tmpdata.neighbors_z, datalist[i2].neighbors_z[subsets[i]]])
        tmpdata.X = np.concatenate([tmpdata.X, datalist[i2].X[subsets[i]]])
        tmpdata.X5 = np.concatenate([tmpdata.X5, datalist[i2].X5[subsets[i]]])
        tmpdata.coordsvec = np.concatenate([tmpdata.coordsvec, datalist[i2].coordsvec[subsets[i]]])


    file_obj = open('/data/infant/processed/train_data/002-T2w_resampled.cerebrum.N4.rot.subset{0}.obj'.format(i), 'wb')
    pickle.dump(tmpdata, file_obj, protocol=4)

# file_obj = open('/data/infant/processed/train_data/002-T2w_resampled.cerebrum.N4.rot.aug.obj', 'wb')
# pickle.dump(data0, file_obj, protocol=4)


############## prepare data for uncertainty
subj = '032'
var1 = nib.load('/data/infant/processed/test/{0}_var1_10.nii.gz'.format(subj))
var2 = nib.load('/data/infant/processed/test/{0}_var2_10.nii.gz'.format(subj))
var3 = nib.load('/data/infant/processed/test/{0}_var1_10.nii.gz'.format(subj))

size = var1.shape + (3,)
vars = np.zeros(size)

vars[:,:,:,0] = var1.get_data()
vars[:,:,:,1] = var2.get_data()
vars[:,:,:,2] = var3.get_data()

recon = nib.Nifti1Image(vars, affine=var1._affine)
nib.save(recon, '/data/infant/processed/test/{0}_vars_10.nii.gz'.format(subj))