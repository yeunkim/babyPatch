import numpy as np
# import nibabel as nib
from sklearn.cluster import KMeans

def normalize_mri(input, mask, ks, init=None):
    # dims = input.shape
    input = input.ravel()
    idxs = np.where(mask.ravel() > 0)
    nonzeros = input[idxs]

    if init:
        k = KMeans(ks, init=np.expand_dims(np.array(init), 1)).fit(np.expand_dims(nonzeros, 1))
        centers = []
        for i in range(len(k.cluster_centers_)):
            cluster_size = np.count_nonzero(k.labels_ == i)
            centers.append(cluster_size)
        idx = np.argsort(centers)
    else:
        k = KMeans(ks).fit(np.expand_dims(nonzeros, 1))
        centers = []
        for i in range(len(k.cluster_centers_)):
            cluster_size = np.count_nonzero(k.labels_ == i)
            centers.append(cluster_size)
        idx = np.argsort(centers)
    wm_mean = k.cluster_centers_[idx[-1]][0]
    std = np.std(nonzeros)

    return wm_mean, std