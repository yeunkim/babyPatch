import nibabel as nib
import numpy as np
import pickle

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

def render_uncert_imgs(fns, var, niis, subjs, numchannels, iterations):
    for f in np.arange(len(fns)):
        file_obj = open(fns[f], 'rb')
        test1 = pickle.load(file_obj)
        size = test1.dataOrigShape
        var1 = np.zeros(size)
        var2 = np.zeros(size)
        var3 = np.zeros(size)

        for i in np.arange(var[f].shape[0]):
            # idxs = np.unravel_index(test1.indices[i], test1.dataOrigShape)
            var1.ravel()[i] = var[f][i, 0]
            var2.ravel()[i] = var[f][i, 1]
            var3.ravel()[i] = var[f][i, 2]
        del test1
        recon = nib.Nifti1Image(var1, affine=niis[f])
        nib.save(recon,
                 filename='/data/infant/processed/test/{0}_var1_i{2}_{1}ch_en_{3}.nii.gz'.format(subjs[f], numchannels,
                                                                                                iterations, int(f+1)))
        recon = nib.Nifti1Image(var2, affine=niis[f])
        nib.save(recon,
                 filename='/data/infant/processed/test/{0}_var2_i{2}_{1}ch_en_{3}.nii.gz'.format(subjs[f], numchannels,
                                                                                                iterations, int(f+1)))
        recon = nib.Nifti1Image(var3, affine=niis[f])
        nib.save(recon,
                 filename='/data/infant/processed/test/{0}_var3_i{2}_{1}ch_en_{3}.nii.gz'.format(subjs[f], numchannels,
                                                                                                iterations, int(f+1)))

        sizevar = size + (3,)
        vars = np.zeros(sizevar)

        vars[:, :, :, 0] = var1
        vars[:, :, :, 1] = var2
        vars[:, :, :, 2] = var3

        recon = nib.Nifti1Image(vars, affine=niis[f])
        nib.save(recon,
                 '/data/infant/processed/test/{0}_vars_i{2}_{1}ch_en_{3}.nii.gz'.format(subjs[f], numchannels, iterations,
                                                                                       int(f+1)))