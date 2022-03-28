from models.cnn_morphs import dilation_nn, erosion_nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import skimage.morphology as morph
from skimage.feature import canny
from diceCoeff import diceCoeff, diceCoeff_torch
import numpy as np
import math
import torch
from RL.optimize import optimize_model
from Dataset.testDataset import ToTensor, testDataset, shapes
from torch.utils.data import DataLoader
from skimage.measure import regionprops
from scipy.ndimage import affine_transform
from evaluation import maxhd

class ReplayMemory(object):

    def __init__(self, capacity, Transition):
        self.memory = deque([],maxlen=capacity)
        self.Transition = Transition

    def push(self, image, label, action, next_image, next_label, reward):
        """Save a transition"""
        self.memory.append(self.Transition(image, label, action, next_image, next_label, reward))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Solver(object):
    def __init__(self, img, label, batch_size = 1000,  patch_size=10, channels=1, gamma = 0.9, eps_start=0.999,
                 eps_end=0.05, eps_decay=200,):
        self.img = img
        self.label = label
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.channels = channels
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

        self.actions = shapes()
        self.n_actions = len(self.actions.aff_matrices)
        self.dil_policy_net = dilation_nn(self.patch_size, self.patch_size, self.n_actions,
                                          channels).cuda()
        self.dil_target_net = dilation_nn(self.patch_size, self.patch_size, self.n_actions,
                                          channels).cuda()
        self.dil_target_net.load_state_dict(self.dil_policy_net.state_dict())
        self.dil_target_net.eval()

        self.dil_optimizer = optim.Adam(self.dil_policy_net.parameters())

        self.data = testDataset(self.img, self.label, ToTensor())
        self.dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True, num_workers=0, drop_last=False)

    def dilate(self, action, cluster_img):
        cluster_img = cluster_img.cpu()

        cluster_img_new = affine_transform(cluster_img[:,0,:].view(11,11), self.actions.aff_matrices[action],
                                       order=0, mode='constant', cval=0) #output_shape=cluster_img.shape,
        done = False
        cluster_img_new = torch.Tensor(cluster_img_new)
        cluster_img_new = cluster_img_new.view((-1, 1, cluster_img.shape[-2], cluster_img.shape[-1]))
        # cluster_img = cluster_img.view((-1, 1, cluster_img.shape[-2], cluster_img.shape[-1]))
        label_hat = torch.cat((cluster_img_new , cluster_img[:,:-1]), 1)
        if torch.sum(torch.abs(label_hat[:,0] - label_hat[:,1])) == 0:
        # if (action+1) == self.n_actions:
            done = True
        return label_hat, done

    def erode(self,erosion_actions0, cluster_image):
        ''' Have all voxels at edge stacked vertically '''
        edge = canny(cluster_image)
        edge_idxs = np.where(edge > 0)
        cluster_image[edge_idxs] = erosion_actions0

        return cluster_image

    # create reward function
    def calc_reward(self,cluster_image, ground_truth,action, oldmetric):
        ground_truth = ground_truth[:,0,:]
        label_hat = cluster_image[:,0,:]
        dsc = diceCoeff_torch(label_hat, ground_truth, 1)
        hd = maxhd(label_hat, ground_truth) + 2
        if hd == 0:
            hd=1
        hdinv = 1 / hd
        print('Dice: ', dsc, 'HD: ', hd)
        # vol = np.sum(cluster_image)
        # edge = canny(cluster_image)
        # perim = np.sum(edge)
        # v2p = vol / perim
        metric = (1*dsc) + (0.5*hdinv)
        uniqueness = torch.sum(torch.abs(cluster_image[:,0,:] - cluster_image[:,-1,:])) - 1
        if uniqueness > 0:
            uniqueness = 0
        else:
            uniqueness = -1
        # if done and (metric < 0.4) and action != (self.n_actions-1):
        #     metric = torch.tensor(-1, dtype=torch.float)
        # if done and (metric > 0.4 or metric == 0.4):
        #     metric = torch.tensor(0.6, dtype=torch.float)
        # if (dsc > 0.9): # and action == (self.n_actions - 1)
        #     metric = metric * 3
        # elif (dsc > 0.8) : # and action == (self.n_actions - 1)
        #     metric = metric * 2
        # elif (dsc > 0.3) and action == (self.n_actions-1):
        #     metric = metric *2 # torch.tensor(1, dtype=torch.float)

        # if dsc < 0.1: #(torch.sum(ground_truth) > 7) and
        #     metric = torch.tensor(-1, dtype=torch.float)
        reward = metric + uniqueness

        if (metric - oldmetric) <0:
            reward = -1*(torch.abs(reward) + torch.abs(oldmetric))

        return reward, metric

    def select_action(self,state, n_actions):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        image, label = state
        if (sample > eps_threshold):
            with torch.no_grad():
                print('Select action using policy net')
                return self.dil_policy_net(image, label).max(1)[1]
        else:
            print('Select action using randomization')
            return torch.randint(0, n_actions, (image.shape[0],)).cuda()

    def select_action_eval(self,state):
        image, label = state
        with torch.no_grad():
            print('Select action using policy net')
            return self.dil_target_net(image, label).max(1)[1]


    def train(self, target_update=50, num_episodes=10, memory_cap = 1000):

        self.steps_done = 0
        self.dil_transition = namedtuple('Transition',
                                ('image', 'label', 'action', 'next_image', 'next_label', 'reward'))

        dilation_memory = ReplayMemory(memory_cap, self.dil_transition)

        n =0
        for episode in range(num_episodes):
            print('Starting episode: ', episode+1)
            for i, (sample, idx) in enumerate(self.dataloader):
                image = sample['image'].cuda()
                label = sample['label'].cuda()
                init = np.zeros( label.shape)
                label_hat = np.concatenate((self.actions.circle, init, init), axis= 1)
                label_hat = torch.from_numpy(label_hat.astype('float32'))
                label_hat = torch.autograd.Variable(label_hat, requires_grad=False).cuda()
                state = (image, label_hat)
                metric = 0
                finished = False
                while not finished:
                    print('Episode: ', episode + 1)
                    # select dilation action
                    dil_action = self.select_action(state, self.n_actions)
                    print('Action: ', dil_action)
                    # take dilation action
                    label_hat2, done = self.dilate(dil_action, label_hat)
                    reward, metric = self.calc_reward(label_hat2, label,dil_action, metric)
                    print('Reward: ', reward)
                    print('label_hat2: ', label_hat2[:, 0, :])
                    print('gt: ', label)
                    # optimize model
                    optimize_model(dilation_memory, 1000, self.dil_transition, self.gamma, self.dil_optimizer,
                                   self.dil_policy_net, self.dil_target_net)
                    n+=1
                    label_hat2 = label_hat2.cuda()
                    newimage = image
                    if done == True:
                        newimage = None
                        label_hat2 = None
                        print('Finished batch ', i +1)
                        finished = True

                    dilation_memory.push(image, label_hat, dil_action, newimage, label_hat2, reward)

                    if n % target_update == 0:
                        self.dil_target_net.load_state_dict(self.dil_policy_net.state_dict())

                    newstate = (image, label_hat2)
                    state = newstate
                    label_hat = label_hat2


    def test(self, images, labels):
        data = testDataset(images, labels, ToTensor())
        dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=0,
                                     drop_last=False)
        label_hats = torch.zeros((len(images), 11, 11))
        for i, (sample, idx) in enumerate(dataloader):
            image = sample['image'].cuda()
            label = sample['label'].cuda()
            init = np.zeros(label.shape)
            label_hat = np.concatenate((self.actions.circle, init, init), axis=1)
            label_hat = torch.from_numpy(label_hat.astype('float32'))
            label_hat = torch.autograd.Variable(label_hat, requires_grad=False).cuda()
            state = (image, label_hat)
            finished = False
            metric = 0
            n = 0
            metrics = []
            while not finished:
                print('batch number:', i)
                # select dilation action
                dil_action = self.select_action_eval(state)
                print('Action: ', dil_action)
                # take dilation action
                label_hat2, done = self.dilate(dil_action, label_hat)
                reward, metric = self.calc_reward(label_hat2, label, dil_action, metric)
                metrics.append(metric)

                print('label_hat2: ', label_hat[:, 0, :])
                print('gt: ', label)
                n += 1
                if done == True:
                    finished = True
                    print("\nMetric:", metric)
                state = (image, label_hat.cuda())
                label_hat = label_hat2
                if n > 20:
                    # largest = max(metrics)
                    # if metric == largest:
                    finished = True

            label_hats[i] = label_hat[:, 0, :]

        return label_hats
