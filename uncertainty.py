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

def render_uncert_imgs(fns, var, niis, subjs, numchannels, iterations, pdir, classes=2, pkl=False, suffix=None):
    if suffix:
        suffix = '{0}'.format(suffix)
    else:
        suffix = ''
    for f in np.arange(len(fns)):
        if pkl:
            file_obj = open(fns[f], 'rb')
            test1 = pickle.load(file_obj)
            size = test1.dataOrigShape
        else:
            with h5py.File(fns[f], 'r') as ff:
                size = tuple(ff.attrs['origsize'][:3])

        var1 = var[f][:, 0].reshape(size)
        var2 = var[f][:, 1].reshape(size)
        if classes > 2:
            var3 = var[f][:, 2].reshape(size)

        sizevar = size + (classes,)
        vars = np.zeros(sizevar)

        vars[:, :, :, 0] = var1
        vars[:, :, :, 1] = var2
        if classes > 2:
            vars[:, :, :, 2] = var3

        recon = nib.Nifti1Image(vars, affine=niis[f])
        nib.save(recon,
                 '/{4}/{0}_vars_i{2}_{1}ch_{3}.nii.gz'.format(subjs[f], numchannels, iterations, suffix, pdir))

def select_best_model(gm2wm, csf2wm, iterations, initoutputs_alliterations, initimodels, initinterfeatimgs_alliterations,
                      valinitinterfeatimgs = None, validation=False):
    maes = []
    for i in np.arange(iterations):
        mae = 0
        for j in np.arange(len(initoutputs_alliterations[i])):
            nii = nib.load(initoutputs_alliterations[i][j])
            data = nii.get_fdata()
            wm = np.sum(data == 1)
            gm = np.sum(data == 2)
            csf = np.sum(data == 3)
            mae = (np.abs((gm / wm) - gm2wm) + np.abs((csf / wm) - csf2wm)) / 2
            mae += mae
            del data
        maes.append(mae / len(initoutputs_alliterations))
    smallestidx = np.argmin(maes)
    print(initimodels[smallestidx])
    initinterfeatimgs_for_refstage = initinterfeatimgs_alliterations[smallestidx] #[:-len(valinitinterfeatimgs)]
    initoutputs_for_refstage = initoutputs_alliterations[smallestidx] #[:-len(valinitinterfeatimgs)]
    # if validation:
    #     valinitinterfeatimgs_for_refstage = initinterfeatimgs_alliterations[smallestidx][-len(valinitinterfeatimgs):]
    #     valinitoutputs_for_refstage = initoutputs_alliterations[smallestidx][-len(valinitinterfeatimgs):]
    #     return initinterfeatimgs_for_refstage,initoutputs_for_refstage,\
    #            valinitinterfeatimgs_for_refstage,valinitoutputs_for_refstage, smallestidx
    # else:
    return initinterfeatimgs_for_refstage, initoutputs_for_refstage, smallestidx