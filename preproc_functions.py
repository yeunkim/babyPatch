import nibabel as nib
import nibabel.processing as proc
import numpy as np

def resample1mm(nii, output, order=3):
    data = nii.get_data()

    shape = data.shape
    affine = np.diag([1,1,1,1])

    for i in range(0,4):
        affine[i][3] = nii.affine[i][3]
    resampled = proc.resample_from_to(nii, [shape, affine], order=order)

    resampled.get_data()[ resampled.get_data() < 0.1] = 0

    recon = nib.Nifti1Image(resampled, affine)
    nib.save(recon, output)
    # return resampled

def mask(nii, output):
    data = nii.get_data()
    affine = nii._affine
    data[ data > 0] = 1
    recon = nib.Nifti1Image(data, affine)
    nib.save(recon, output)
