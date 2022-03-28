
import numpy as np
import nibabel as nib
import subprocess

def makePVCfrac(labelName, prefix):
    label = nib.load(labelName)
    labelmat = label.get_fdata()
    labelmat[label.get_fdata() == 3] = 4
    labelmat[label.get_fdata() == 1] = 3
    labelmat[labelmat == 4] = 1
    recon = nib.Nifti1Image(labelmat.astype(np.float32), affine=label._affine)
    nib.save(recon, '{0}.pvc.frac.nii.gz'.format(prefix))

def run_cortex(prefix):
    cmd = '/data/infant/T2_train_2021/innercorticalmask.sh {0}'.format(prefix)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()

def mask_label(label, prefix):
    nii = nib.load(label)
    labeldat = nii.get_fdata()[:]
    maskdata = nib.load(prefix + '.cortex.wmsubgm.mask.nii.gz').get_fdata()
    labeldat[maskdata == 0] = 0
    labeldat[labeldat != 2] = 0
    labeldat[labeldat == 2] = 255
    gm = labeldat.astype(np.uint8)

    recon = nib.Nifti1Image(gm, affine=nii._affine)
    recon.set_data_dtype(np.uint8)

    nib.save(recon, '{0}.masked.mask.nii.gz'.format(prefix))

def morphOp_label(prefix):
    cmd = '/home/yeunkimlocal/Code/dmorph14c_x86_64-pc-linux-gnu {0}.masked.mask.nii.gz {0}.masked.morph.mask.nii.gz ed1 dc1 dd1 dc1 dd1'.format(
        prefix)
    process = subprocess.Popen(cmd, shell=True)
    process.wait()
    # cmd = '/home/yeunkimlocal/Code/dmorph14c_x86_64-pc-linux-gnu {0}.masked.morph.mask.nii.gz {0}.masked.morph.mask.nii.gz sf'.format(
    #     prefix)
    # subprocess.Popen(cmd, shell=True)

def maskIntersect(prefix, origmask):
    morphmasknii = nib.load('{0}.masked.morph.mask.nii.gz'.format(prefix))
    origmasknii = nib.load(origmask)
    morphmaskmat = morphmasknii.get_fdata()[:]
    morphmaskmat[ origmasknii.get_fdata() == 0 ] = 0
    morphmaskmat[morphmaskmat != 0] = 255
    recon = nib.Nifti1Image(morphmaskmat.astype(np.uint8), affine=morphmasknii._affine)
    nib.save(recon, '{0}.masked.inters.mask.nii.gz'.format(prefix))

def generate_wmsubgm_mask(labelName, prefix, origmask):
    makePVCfrac(labelName, prefix)
    run_cortex(prefix)
    mask_label(labelName, prefix)
    morphOp_label(prefix)
    maskIntersect(prefix, origmask)