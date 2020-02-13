import nibabel as nib
import nibabel.processing as proc
import numpy as np

def resample1mm(nii, output, order=3, mode='constant', ref=None):
    data = nii.get_data()

    if ref:
        resampled = proc.resample_from_to(nii, ref, order=order, mode=mode)

        resampled.get_data()[resampled.get_data() < 0.1] = 0

        # recon = nib.Nifti1Image(resampled, affine)
        nib.save(resampled, output)
    else:
        shape = data.shape
        affine = np.diag([1,1,1,1])

        for i in range(0,4):
            affine[i][3] = nii.affine[i][3]
        resampled = proc.resample_from_to(nii, [shape, affine], order=order, mode=mode)

        resampled.get_data()[ resampled.get_data() < 0.1] = 0

        # recon = nib.Nifti1Image(resampled, affine)
        nib.save(resampled, output)
    # return resampled

def mask(nii, output):
    data = nii.get_data()
    affine = nii._affine
    data[ data > 0] = 1
    recon = nib.Nifti1Image(data, affine)
    nib.save(recon, output)
