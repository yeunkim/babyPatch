import numpy as np
import nibabel as nib

def compute_mean(img, mean, i):
    k = i + 1

    if i == 0:
        mean = img[:]
    else:
        mean = mean + ((img - mean) / k)

    mean[mean < 1] = 0

    return mean
model='syn'
atlasname='/data/infant/T2_train_2021/ANTS_tests/t2_M-CRIB_template_bs.1mm.padded.nii.gz'
atlas = nib.load(atlasname)
mean = np.zeros_like(atlas.get_fdata())

subjs = ['002',
            '010',
            '023',
            '039',
            '056',
            '072',
            '087',
            '108',
            '115',
            '132']

for i in range(len(subjs)):
    imgname = '/data/infant/T2_train_2021/ANTS_tests/{0}_{1}/{0}-C-T1_T2w_synWarped.nii.gz'.format(
        subjs[i], model
    )
    img = nib.load(imgname).get_fdata()
    mean = compute_mean(img, mean, i)

recon = nib.Nifti1Image(mean, atlas._affine)
nib.save(recon, '/data/infant/T2_train_2021/ANTS_tests/averaged.{0}.nii.gz'.format(model))