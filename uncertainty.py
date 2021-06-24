import nibabel as nib
import numpy as np
import pickle
import h5py

def compute_var_mean(label_OHE, mean, var, i):
    k = i + 1
    for f in np.arange(len(label_OHE)):
        if i == 0:
            mean.append(label_OHE[f])
        elif i == 1:
            var.append(((1/k)*(label_OHE[f]-mean[f])**2))
            mean[f] = mean[f] + ((label_OHE[f]- mean[f]) / k)
        else:
            var[f] = (((k - 2) / (k - 1)) * (var[f])) + ((1 / k) * (label_OHE[f] - mean[f]) ** 2)
            mean[f] = mean[f] + ((label_OHE[f] - mean[f]) / k)
    return mean, var

def compute_var_mean_individual(label_OHE, mean, var, i):
    k = i + 1
    if i==0:
        mean = label_OHE[0]
    else:
        var = (((k-2)/(k-1))*(var)) + ((1/k)*(label_OHE[0]-mean)**2)
        mean = mean + ((label_OHE[0] - mean) / k)

    return mean, var

def render_uncert_imgs(fns, var, niis, subjs, numchannels, iterations, mean=None, pkl=False, suffix=None):
    if suffix:
        suffix = '_{0}'.format(suffix)
    else:
        suffix = ''
    for f in np.arange(len(fns)):
        #file_obj = open(fns[f], 'rb')
        #test1 = pickle.load(file_obj)
        if pkl:
            file_obj = open(fns[f], 'rb')
            test1 = pickle.load(file_obj)
            size = test1.dataOrigShape
        else:
            with h5py.File(fns[f], 'r') as ff:
                size = tuple(ff.attrs['origsize'][:3])

        #var1 = np.zeros(size)
        #var2 = np.zeros(size)
        #var3 = np.zeros(size)

        #for i in np.arange(var[f].shape[0]):
            # idxs = np.unravel_index(test1.indices[i], test1.dataOrigShape)
        var1 = var[f][:, 0].reshape(size)
        var2 = var[f][:, 1].reshape(size)
        var3 = var[f][:, 2].reshape(size)
        #del test1
        recon = nib.Nifti1Image(var1, affine=niis[f])
        nib.save(recon,
                 filename='/data/infant/variance/{0}_var1_i{2}_{1}ch_en_{3}{4}.nii.gz'.format(subjs[f], numchannels,
                                                                                                iterations, f, suffix))
        recon = nib.Nifti1Image(var2, affine=niis[f])
        nib.save(recon,
                 filename='/data/infant/variance/{0}_var2_i{2}_{1}ch_en_{3}{4}.nii.gz'.format(subjs[f], numchannels,
                                                                                                iterations, f, suffix))
        recon = nib.Nifti1Image(var3, affine=niis[f])
        nib.save(recon,
                 filename='/data/infant/variance/{0}_var3_i{2}_{1}ch_en_{3}{4}.nii.gz'.format(subjs[f], numchannels,
                                                                                                iterations, f, suffix))

        sizevar = size + (3,)
        vars = np.zeros(sizevar)

        vars[:, :, :, 0] = var1
        vars[:, :, :, 1] = var2
        vars[:, :, :, 2] = var3

        recon = nib.Nifti1Image(vars, affine=niis[f])
        nib.save(recon,
                 '/data/infant/variance/{0}_vars_i{2}_{1}ch_en_{3}{4}.nii.gz'.format(subjs[f], numchannels, iterations,
                                                                                       f, suffix))

        if mean is not None:
            means = np.zeros(sizevar)
            means[:, :, :, 0] = mean[f][:, 0].reshape(size)
            means[:, :, :, 1] = mean[f][:, 1].reshape(size)
            means[:, :, :, 2] = mean[f][:, 2].reshape(size)

            recon = nib.Nifti1Image(means, affine=niis[f])
            nib.save(recon, '/data/infant/variance/{0}_means_i{2}_{1}ch_en_{3}{4}.nii.gz'.format(subjs[f], numchannels,
                                                                                                    iterations, f, suffix))