import numpy as np

def diceCoeff(test, truth, labelnum):

    tmp_test0 = np.zeros_like(test)
    tmp_test0[test == labelnum] = 1.
    tmp_test = np.empty_like(test)
    tmp_test.fill(-1)
    tmp_test[ test == labelnum] = 1.

    tmp_truth = np.zeros_like(truth)
    tmp_truth[truth == labelnum] = 1.

    intersect = float(len(np.where(tmp_test.ravel() == tmp_truth.ravel())[0]))
    denom = np.sum(tmp_test0 + tmp_truth)

    return (2.*intersect)/denom

