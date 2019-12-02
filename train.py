import run_two_stage_cnn
import pickle
import nibabel as nib
import MRDataSet2_noupsample
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import itertools
import numpy as np
import data_preproc_noupsample
d=1


#### fn = All training data object
fn = '/data/infant/vae_objs/002_T2wT1w_augaugbs_1d_3c_p3_yzxline5_nzy5_7_ants_norm.obj'
#### Load obj = Single original training data object
obj1ch = '/data/infant/vae_objs/002_T2wT1w_bs_1d_3c_p3_yzxline5_nzy5_7_ants_norm.obj'
#### image affine matrix
nii = nib.load('/data/infant/processed/002-C-T1_T2w.bst.bse.N3.nn.nii.gz')._affine
#### init output names
initinterfeatimg = '/data/infant/processed/test/002_3channel_test_x_noz.nii.gz'
initoutput = '/data/infant/processed/test/002_3channel_test_predicted_noz.nii.gz'
#### cerebrum mask
cerebrum_mask = '/data/T1002/T1002-5/002-skullstripped_anat.cerebrum.mask.nii'
#### label file
label = '/data/T1002/T1002-5/T1002_edit.6.label.nii.gz'
#### 3-channel interm obj
obj3ch = '/data/infant/vae_objs/002_T2w_light_3c_unraveledidx_p3_yzxline5_nzy5_ants_norm_3channel_noz.obj'
#### refine output names
refineinterfeatimg = '/data/infant/processed/test/002_3channel_test_x_noz_b.nii.gz'
refineoutput = '/data/infant/processed/test/002_3channel_test_predicted_noz_b.nii.gz'
#### param settings
numchannels = 3
multiinput=False
threedim=False

###### save models? #####
savemodel = True
model1output = '/data/infant/vae_objs/002_1d_T2wT1w_bs_augaug_ants_norm_k512n_e5_lr5e4_3c_p3_yzxline5_nzy5_7_noz_3channels.obj'
model2output = '/data/infant/vae_objs/002_1d_T2wT1w_bs_augaug_ants_norm_k512n_e5_lr5e4_3c_p3_yzxline5_nzy5_7_noz_3channels.obj'


################################################################################################
############ START ################################################
starttime = time.time()

solver = run_two_stage_cnn.Solver(fn, epoch=5, lr=5e-4, f_dim=3, batch_size=1000, in_features=1, labels=3, shuffle=True, channels=1)

solver.train()

if savemodel:
    model_obj = open(model1output, 'wb')
    pickle.dump(solver, model_obj, protocol=4)

file_obj = open(obj1ch, 'rb')
test1 = pickle.load(file_obj)

testdata = MRDataSet2_noupsample.MRDataSet(pkl_file=obj1ch,
                                transform=transforms.Compose([
                                    MRDataSet2_noupsample.ToTensor(multiinput=multiinput, threedim=threedim)
                                ]), multiinput=multiinput, threedim=threedim)

dataloader = DataLoader(testdata, batch_size=10000, shuffle=False,
                             num_workers=0, drop_last=False)

size = testdata.dataset.dataOrigShape

#### Generate intermediate feature image
X = solver.test(dataloader, size, test1.indices, test1.dataUpsampledShape, test1.patchsize)

print('002 first model prediction finished.')

b = np.asarray(list(itertools.chain.from_iterable(solver.xhats)))
values = np.zeros([b.shape[0], numchannels])
for i in np.arange(len(b)):
    values[i] = b[i].data.cpu().numpy()

size4d = size + (numchannels,)
Y = np.zeros(size4d)
for i in np.arange(len(b)):
    idxs = np.unravel_index(test1.indices[i], size)
    Y[idxs] = values[i]

recon = nib.Nifti1Image(Y, affine=nii)

#### print out the intermediate feature image
nib.save(recon, filename=initinterfeatimg)

#### print out the initialization stage output
recon = nib.Nifti1Image(X, affine=nii)
nib.save(recon, filename=initoutput)

del solver
del test1
del testdata
del dataloader

##### perform image pre-processing on the intermediate feature image
data0 = data_preproc_noupsample.imagepatches(
                    fname=initinterfeatimg,
                    mask=cerebrum_mask,
                    label= label,
    patchsize=d, upsample=d, gm=2, wm=1, csf=3, num_classes=4, order=3, pad=5, ysize=5, zsize=1,
                                             channels=3
                                             )

file_obj = open(obj3ch, 'wb')
pickle.dump(data0, file_obj, protocol=4)

fn = obj3ch
solver = run_two_stage_cnn.Solver(fn, epoch=5, lr=5e-4, f_dim=3, batch_size=1000, in_features=1, labels=3, shuffle=True, channels=3)

solver.train()

if savemodel:
    model_obj = open(model2output, 'wb')
    pickle.dump(solver, model_obj, protocol=4)

###### load intermediate feature image

file_obj = open(obj3ch, 'rb')
test1 = pickle.load(file_obj)

testdata = MRDataSet2_noupsample.MRDataSet(pkl_file=obj3ch,
                                transform=transforms.Compose([
                                    MRDataSet2_noupsample.ToTensor(multiinput=multiinput, threedim=threedim)
                                ]), multiinput=multiinput, threedim=threedim)

dataloader = DataLoader(testdata, batch_size=10000, shuffle=False,
                             num_workers=0, drop_last=False)

#### Refinement Stage prediction

X = solver.test(dataloader, size, test1.indices, test1.dataUpsampledShape, test1.patchsize)

print('002 second model prediction finished.')

#### Save outputs

b = np.asarray(list(itertools.chain.from_iterable(solver.xhats)))
values = np.zeros([b.shape[0], numchannels])
for i in np.arange(len(b)):
    values[i] = b[i].data.cpu().numpy()

Y = np.zeros(size4d)
for i in np.arange(len(b)):
    idxs = np.unravel_index(test1.indices[i], size)
    Y[idxs] = values[i]

##### print out the intermediate feature image from refinement stage
recon = nib.Nifti1Image(Y, affine=nii)
nib.save(recon, filename=refineinterfeatimg)

#### print out the refinement stage output
recon = nib.Nifti1Image(X, affine=nii)
nib.save(recon, filename=refineoutput)

elapsed = time.time() - starttime