import numpy as np
import nibabel as nib
import data_preproc_onestage
import pickle

def generate_mask(prefix,obj):
    slicesused = nib.load(prefix + '.nii.gz')
    file_obj = open(obj, 'rb')
    data = pickle.load(file_obj)
    x = np.zeros(data.dataOrigShape)
    x.ravel()[data.indices] = 255
    recon = nib.Nifti1Image(x.reshape(data.dataOrigShape).astype(np.uint8), affine=slicesused._affine)
    del data
    fname = prefix + '.mask.nii.gz'
    nib.save(recon, fname)
    return fname

def generate_obj_files(obj,img,mask,label,numchannels=4,num_classes=3, pad=5):
    ext = 'obj'
    data0 = data_preproc_onestage.imagepatches(
        fname=img,
        mask=mask,
        label=label,
        gm=2, wm=1, csf=3, num_classes=num_classes, channels=numchannels,normalize=False,
        masklabel=True, pad=pad)
    file_obj = open('{0}.{1}'.format(obj, ext), 'wb')
    pickle.dump(data0, file_obj, protocol=4)
    file_obj.close()