''' Scratch code for testing dilation erosion segmentation method '''


import numpy as np
from skimage.draw import disk
from run.run_morph import Solver
import random
import torch
from scipy.ndimage import affine_transform
from Dataset.testDataset import shapes

''' Reward function could be dice score, V:Perimeter ratio '''

''' Actions could be (1) morphological structure element, (2) iterations, (3) '''

''' Q matrix/function could be '''


# create a 2D image with circle
# groundTruth = np.zeros((11,11), dtype=np.uint8)
# rr, cc = disk((1, 5), 6)
# groundTruth[rr, cc] = 1
total = 30
# do affine transformations instead
circle = np.zeros((11,11), dtype=np.uint8)
rr, cc = disk((5, 5), 4)
circle[rr,cc] = 1

testImages = np.zeros((total,11,11))
testLabels = np.zeros((total,11,11))
xfm = shapes()

# for i in range(0,6):
#     testImage = np.zeros((11, 11))
#     testLabel = affine_transform(circle, xfm.aff_matrices[i],
#                                        order=0, mode='nearest')
#     testImage[testLabel > 0] = np.random.randn(np.count_nonzero(testLabel))
#     testImages[i] = testImage
#     testLabels[i] = testLabel

for i in range(0,total):
    testImage = np.zeros((11,11))
    rr, cc = disk((random.randrange(0,6), random.randrange(0,6)), random.randrange(3,8), shape=(11,11))
    testImage[rr, cc] = 1
    num = np.count_nonzero(testImage)
    testImage[testImage>0] = np.random.randn(num)
    testImages[i] = testImage
    testLabel = np.zeros((11,11))
    testLabel[testImage != 0] = 1
    testLabels[i] = testLabel

solver = Solver(testImages, testLabels, batch_size= 1, eps_decay=1000, eps_start=1.0, eps_end=0.001)
solver.train(target_update=50, num_episodes=600,  memory_cap=10000000)
#
torch.save(solver.dil_policy_net.state_dict(), '/data/infant/checkpoints/RL_rand4.pth')

# solver.dil_target_net.load_state_dict(torch.load('/data/infant/checkpoints/RL_rand3.pth'))
# label_hats = solver.test(testImages, testLabels)