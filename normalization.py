import numpy as np
# import nibabel as nib
from sklearn.cluster import KMeans

def normalize_mri(input, mask, ks, init):
    # dims = input.shape
    input = input.ravel()
    idxs = np.where(mask.ravel() > 0)
    nonzeros = input[idxs]

    k = KMeans(ks, init=np.expand_dims(np.array(init), 1)).fit(np.expand_dims(nonzeros, 1))
    wm_mean = k.cluster_centers_[0][0]
    std = np.std(nonzeros)

    return wm_mean, std