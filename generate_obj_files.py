import numpy as np
import nibabel as nib
import data_preproc_onestage
import pickle

def generate_mask(img):
    slicesused = nib.load(img)
    slicesused_data = slicesused.get_fdata()[:, :, :, 0]
    slicesused_data[slicesused_data > 0] = 255
    recon = nib.Nifti1Image(slicesused_data.astype(np.uint8), slicesused._affine)
    fname = img.split('.')[0] + '_mask.nii.gz'
    nib.save(recon, fname)
    return fname

def generate_obj_files(obj,img,mask,label,numchannels=4,num_classes=3, pad=5):
    ext = 'obj'
    data0 = data_preproc_onestage.imagepatches(
        fname=img,
        mask=mask,
        label=label,
        gm=2, wm=1, csf=3, num_classes=num_classes, channels=numchannels,
        masklabel=True, pad=pad)
    file_obj = open('{0}.{1}'.format(obj, ext), 'wb')
    pickle.dump(data0, file_obj, protocol=4)
    file_obj.close()