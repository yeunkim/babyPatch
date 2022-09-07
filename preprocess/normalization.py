import numpy as np
from collections import Counter
from sklearn.cluster import KMeans

def normalize_mri(input, mask, ks, init=None):
    input = input.ravel()
    idxs = np.where(mask.ravel() > 0)
    nonzeros = input[idxs]

    if init:
        k = KMeans(ks, init=np.expand_dims(np.array(init), 1)).fit(np.expand_dims(nonzeros, 1))
    else:
        k = KMeans(ks).fit(np.expand_dims(nonzeros, 1))
    counts = Counter(k.labels_)
    label = max(counts, key=counts.get)
    m = np.mean(nonzeros[k.labels_ == label])
    s = np.std(nonzeros[k.labels_ == label])

    return m, s
