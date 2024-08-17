import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import medpy.metric
import nibabel as nib

def hd(test, truth):

    results = []
    for c in range(1,4):
        test_tmp = np.zeros_like(test)
        test_tmp[test == c] = 1

        truth_tmp = np.zeros_like(truth)
        truth_tmp[truth == c] =1

        ## create edges
        # test_edges = cv2.Canny(test_tmp.astype(np.uint8), 0, 0.1)
        # truth_edges = cv2.Canny(truth_tmp.astype(np.uint8), 0, 0.1)

        ## comput hd
        result = medpy.metric.hd(test_tmp, truth_tmp)
        results.append(result)

    return results
#
def hd95(test, truth):

    results = []
    for c in range(1, 4):
        test_tmp = np.zeros_like(test)
        test_tmp[test == c] = 1

        truth_tmp = np.zeros_like(truth)
        truth_tmp[truth == c] = 1

        ## create edges
        # test_edges = cv2.Canny(test_tmp.astype(np.uint8), 0, 0.1)
        # truth_edges = cv2.Canny(truth_tmp.astype(np.uint8), 0, 0.1)

        ## comput hd
        result = medpy.metric.hd95(test_tmp, truth_tmp)
        results.append(result)

    return results

def sim(labels):
    dims = labels.shape
    diff = 0
    for i in np.arange(dims[0]):
        for j in np.arange(dims[1]):

            if labels[i,j] != 0:
                c = labels[i,j]
                diff += int((labels[i + 1, j] != c) & (labels[i + 1, j] != 0))
                diff += int((labels[i - 1, j] != c) & (labels[i - 1, j] != 0))
                diff += int((labels[i, j + 1] != c) & (labels[i, j + 1] != 0))
                diff += int((labels[i, j - 1] != c) & (labels[i, j - 1] != 0))

    total = np.count_nonzero(labels)
    avgdiff = diff/total

    return avgdiff

subjs = ['025', '031', '036', '027', '039', '040']
# subjs = ['025']
slicenumsls =[[91, 106, 130],[85,99,127],[97, 115, 140], [93, 110, 135],[77, 93, 119],[88, 102, 126]]
slicenumsls2 = [[125, 123], [101, 112] , [123, 116],[134, 118],[108, 121],[125, 118]]

simscores_gt = []
simscores_dn = []
simscores_cnn = []
simscores_ibeat = []
simscores_idn = []

for s in range(0,6):
    subj = subjs[s]
    slicenums = slicenumsls[s]

    dn = nib.load(
        '/data/infant/processed/test/{0}_3channel_test_noz.nii.gz'.format(
            subj))

    idn = nib.load('/data/infant/processed/test/{0}_predicted_test_x_noz.nii.gz'.format(subj))

    cnn = nib.load('/data/infant/processed/test/vq-vae_{0}_dataloader_cycle_nzy5_noz_modelsaved.nii.gz'.format(subj))

    ibeat = nib.load('/data/T1{0}-5/T1{0}-5-reoriented-strip-seg.nii.gz'.format(subj))

    label = nib.load('/data/Infant_Data/{0}/{0}.ED.label.nii.gz'.format(subj))

    for i in range(0, 3):
        dndat = dn.get_data()[:, :, slicenums[i]]
        cnndat = cnn.get_data()[:, :, slicenums[i]]
        ibeatdat = ibeat.get_data()[:, :, slicenums[i]]
        truthdat = label.get_data()[:, :, slicenums[i]]
        idndat = idn.get_data()[:, :, slicenums[i]]

        simscore_dn = sim(dndat)
        simscore_cnn = sim(cnndat)
        simscore_ibeat = sim(ibeatdat)
        simscore_gt = sim(truthdat)
        simscore_idn = sim(idndat)

        simscores_dn.append(simscore_dn)
        simscores_cnn.append(simscore_cnn)
        simscores_ibeat.append(simscore_ibeat)
        simscores_gt.append(simscore_gt)
        simscores_idn.append(simscore_idn)

    for i in range(0,2):
        slicenums = slicenumsls2[s]
        if i ==0:
            dndat = dn.get_data()[:,slicenums[i],:]
            cnndat = cnn.get_data()[:, slicenums[i], :]
            ibeatdat = ibeat.get_data()[:, slicenums[i], :]
            truthdat = label.get_data()[:, slicenums[i], :]
            idndat = idn.get_data()[:, slicenums[i], :]
        else:
            dndat = dn.get_data()[slicenums[i], :, :]
            cnndat = cnn.get_data()[slicenums[i], :, :]
            ibeatdat = ibeat.get_data()[slicenums[i], :, :]
            truthdat = label.get_data()[slicenums[i], :, :]
            idndat = idn.get_data()[slicenums[i], :, :]

        simscore_dn = sim(dndat)
        simscore_cnn = sim(cnndat)
        simscore_ibeat = sim(ibeatdat)
        simscore_gt = sim(truthdat)
        simscore_idn = sim(idndat)

        simscores_dn.append(simscore_dn)
        simscores_cnn.append(simscore_cnn)
        simscores_ibeat.append(simscore_ibeat)
        simscores_gt.append(simscore_gt)
        simscores_idn.append(simscore_idn)


import csv
# with open('/data/simscores_dn.csv', 'w') as f:
#     # for i in range(0, len(simscores_dn)):
#     writer = csv.writer(f, delimiter='\n')
#     writer.writerow(np.asarray(simscores_dn))
#
# with open('/data/simscores_cnn.csv', 'w') as f:
#     # for i in range(0, len(simscores_dn)):
#     writer = csv.writer(f, delimiter='\n')
#     writer.writerow(np.asarray(simscores_cnn))
#
# with open('/data/simscores_ibeat.csv', 'w') as f:
#     # for i in range(0, len(simscores_dn)):
#     writer = csv.writer(f, delimiter='\n')
#     writer.writerow(np.asarray(simscores_ibeat))
#
# with open('/data/simscores_gt.csv', 'w') as f:
#     # for i in range(0, len(simscores_dn)):
#     writer = csv.writer(f, delimiter='\n')
#     writer.writerow(np.asarray(simscores_gt))
#
with open('/data/simscores_idn.csv', 'w') as f:
    # for i in range(0, len(simscores_dn)):
    writer = csv.writer(f, delimiter='\n')
    writer.writerow(np.asarray(simscores_idn))