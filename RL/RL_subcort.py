import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import math

## load data
# from RL_inference import runRL_inference

## process and save out data
image_dir = '/data/infant/T2_train_2021/'
image_file_suffix = '-C-T1_T2w.1mm.bse.N4.full+mask_s3.nii.gz'
# mask_file_suffix = '-C-T1_T2w_affine.mask.nii.gz'
# image_dir = '/data/infant/intermediate_nii/'
# image_file_suffix = '_4ch_initinterfeatimg_e10_i4_2021_10slices_sparse1.nii.gz'
mask_file_suffix = '-C-T1_T2w_affine.mask.nii.gz'
label_dir = '/data/infant/t2traindata_labels_handedit_YK_05072020'
label_file_suffix = '-C-T1.T2w.final.label.nii.gz'
save_dir = '/data/infant/objects/'
suffix = "_subcortex"
subj = '010'
subjs = ['010', '023'] # '115','132' , '087', '072', '056', '039','023','010', '002' # '108'
i=0
# maxdims = (64, 56, 41)
# for subj in subjs:
#     mask_dir = '/data/infant/T2_train_2021/ANTS_tests/{0}toAtlas/'.format(subj)
#     ([xpos, xpos_end],
#      [ypos, ypos_end],
#      [zpos, zpos_end]) = pickle.load(open('{0}/{1}{2}.obj'.format(save_dir,subj,suffix), 'rb'))['bounds']
# #
#     data0 = data_preproc_onestage.imagepatches(
#             fname='{0}/{1}{2}'.format(image_dir,subj,image_file_suffix),
#             setbounds=True,
#             bounds=([xpos, xpos_end + (maxdims[0] - (xpos_end-xpos))],
#                     [ypos, ypos_end + (maxdims[1] - (ypos_end-ypos))],
#                     [zpos, zpos_end+ (maxdims[2] - (zpos_end-zpos))]),
#             mask='{0}/{1}{2}'.format(mask_dir,subj,mask_file_suffix),
#             label='{0}/{1}{2}'.format(label_dir,subj,label_file_suffix),
#             gm=2, wm=1, csf=3, num_classes=4, pad=5, # k_t2=4, k_t2_init=[300, 60, 640, 950], ## use if normalization goes wrong
#             masklabel=True, fnoutput='{0}/{1}{2}'.format(save_dir,subj,suffix),
#             dataNum=i, pkl=True, normalize=False)
#     i+=1

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# keep track of transitions to replay experience later
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, state, action, nextstate, reward):
        """Save a transition"""
        for i in range(len(state)):
            self.memory.append(Transition(state[i], action[i], nextstate[i], reward[i]))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# CNN to estimate next action
class DQN(nn.Module):
    def __init__(self, h, w, channels, outputs):
        super(DQN, self).__init__()

        self.channels = channels
        self.outputs = outputs
        self.f1 = 32
        self.f2 = 64
        self.kw1 = 2
        self.kw2 = 2
        self.kw3 = 2
        self.stride = 1

        self.convX = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f2, self.kw3),
            nn.BatchNorm2d(self.f2)
        )

        self.convY = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f2, self.kw3),
            nn.BatchNorm2d(self.f2)
        )

        self.convZ = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f2, self.kw3),
            nn.BatchNorm2d(self.f2)
        )

        self.labelX = nn.Sequential(
            nn.Conv2d(1, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f2, self.kw3),
            nn.BatchNorm2d(self.f2)
        )

        self.labelY = nn.Sequential(
            nn.Conv2d(1, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f2, self.kw3),
            nn.BatchNorm2d(self.f2)
        )

        self.labelZ = nn.Sequential(
            nn.Conv2d(1, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f2, self.kw3),
            nn.BatchNorm2d(self.f2)
        )

        def conv2d_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, self.kw1), self.kw2), self.kw3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, self.kw1), self.kw2), self.kw3)
        linear_input_size = convw * convh * self.f2 * 3

        h1=256
        h2=100
        self.head = nn.Sequential(nn.Linear(linear_input_size, h1),
                                  # nn.BatchNorm1d(h2),
                                  nn.LeakyReLU(0.1, True),
                                  nn.Linear(h1, outputs)
                                  )


    def forward(self, x, y, z ,label_x, label_y, label_z, ):
        x = self.convX(x)
        x = x.view(x.size(0), -1)

        y = self.convY(y)
        y = y.view(y.size(0), -1)

        z = self.convZ(z)
        z = z.view(z.size(0), -1)

        labelx = self.labelX(label_x)
        labely = self.labelY(label_y)
        labelz = self.labelZ(label_z)
        labelx = labelx.view(labelx.size(0), -1)
        labely = labely.view(labely.size(0), -1)
        labelz = labelz.view(labelz.size(0), -1)

        x = x*labelx
        y = y*labely
        z = z*labelz

        out = torch.cat((x,y,z),1)

        return self.head(out.view(out.size(0), -1))

# unravel idxs
def unravel_arrs(indices, shape):
    dzdy = shape[1]*shape[2]
    dz = shape[2]
    k = (indices / dzdy).astype(np.int)
    indices -= dzdy * k
    j = (indices/dz).astype(np.int)
    indices -= j*dz
    i = (indices).astype(np.int)
    return tuple([i,j,k])

# coloring through the y-plane
# ACTIONS = {     0: [0, 1, 0],
#                 1: [0, -1, 0],
#                 2: [1, 0, 0],
#                 3: [-1, 0, 0],
#                 4: [-1, 1, 0],
#                 5: [1, -1, 0],
#                 6: [-1, -1, 0],
#                 7: [1, 1, 0],
#                 8: [0, 0, 0]
#     }
# ACTIONS = { 0: [0, 1],
#                 1: [0, -1],
#                 2: [1, 0],
#                 3: [-1, 0],
#                 4: [-1, 1],
#                 5: [1, -1],
#                 6: [-1, -1],
#                 7: [1, 1],
#                 9: [0, 0]
#     }

ACTIONS = {     0:[-1, -1, -1],
       1:[-1, -1,  0],
       2:[-1, -1,  1],
       3:[-1,  0, -1],
       4:[-1,  0,  0],
       5:[-1,  0,  1],
       6:[-1,  1, -1],
       7:[-1,  1,  0],
       8:[-1,  1,  1],
       9:[ 0, -1, -1],
       10:[ 0, -1,  0],
       11:[ 0, -1,  1],
       12:[ 0,  0, -1],
       13:[ 0,  0,  1],
       14:[ 0,  1, -1],
       15:[ 0,  1,  0],
       16:[ 0,  1,  1],
       17:[ 1, -1, -1],
       18:[ 1, -1,  0],
       19:[ 1, -1,  1],
       20:[ 1,  0, -1],
       21:[ 1,  0,  0],
       22:[ 1,  0,  1],
       23:[ 1,  1, -1],
       24:[ 1,  1,  0],
       25:[ 1,  1,  1],
       26:[ 0,  0,  0]
    }
ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]
ravel2d = lambda x, y: (y[1] * x[0]) + x[1]

def take_action(coord, action, acts):
    newcoord = coord + acts[tuple((np.arange(0, len(action)), action))]
    # print('newcoord shape:' , newcoord.shape )
    # newcoord[action == 9] = -1
    # newcoord[coord == -1] = -1
    return newcoord


def get_newlabel_and_calc_reward(label, action, label_vec, gt_label_vec, subsetcoords, subsetDataNum,
                                 vislabel, clusternums, seedStreamCount):
    label_new = label[:]
    # change to equals 2 got rid of all the past history
    label_new[ label_new > 0] = 3
    done=np.array([False]*len(action))

    # find coordinates that have already been travelled
    label_ind = np.asarray([label_new[tuple((subsetDataNum, subsetcoords))] > 0]).reshape(-1)
    label_ind = label_ind.astype(np.int)
    # change those seeds to be finished
    # done[ label_ind == 1] = True
    # work with the non-finished seeds
    subsetclusternums = clusternums[label_ind == 1]
    label_new[tuple((subsetDataNum,subsetcoords))] = 1

    # keep track of the stream lengths
    # TODO: change so that it doesn't go back and change history
    seedStreamCount += 1

    # try:
    #     seedlabels = vislabel[tuple((subsetDataNum,subsetcoords))][label_ind == 1]
    #     if len(seedlabels) > 0:
    #         for i, s in enumerate(seedlabels):
    #             vislabel[subsetDataNum[i]][vislabel[subsetDataNum[i]] == s] = subsetclusternums[i]
    #
    # except IndexError:
    #     seedlabels = vislabel[tuple((subsetDataNum, subsetcoords))]
    #     if seedlabels > 0:
    #         vislabel[subsetDataNum[0]][vislabel[subsetDataNum[0]] == seedlabels ] = subsetclusternums[0]

    vislabel[tuple((subsetDataNum,subsetcoords))] = clusternums

    done[action == 27] = True

    if torch.all(subsetcoords == subsetcoords[0]):
        done[done == False] = True
    # intensity difference reward -> eh turn off for now
    # int_reward = (1/(abs(y_img[newcoord] - y_img[coord])))
    # label similarities reward
    # gt_label_vec is the vector that is initialized with the label number of the ground truth label at
    # at the initial region
    label_reward = (torch.abs(label_vec - gt_label_vec)*(-1)) + 2 # + (torch.abs(label_vec - gt_label_vec)*(-2))
    label_reward[label_reward < 2] = 0
    try:
        done[label_reward.cpu() == 0] = True
    except IndexError:
        done = True
    # label_reward[done == True] = 0
    reward = label_reward.type(torch.FloatTensor).cuda()  * seedStreamCount
    # reward = label_reward
    # print('reward:',reward)
    # if np.sum((label_new - label)) ==0:
    #     reward = 0
    return label_new, reward, done, vislabel, seedStreamCount

from collections import Counter
def get_idx_duplicates(array):
    array = np.asarray(array)
    c = Counter(np.asarray(array))
    idxs = [np.where(array == k)[0] for k in c if c[k] > 1]
    uniqRep =[np.where(array == k)[0][0] for k in c if c[k] > 1]
    numRep = [c[array[uniqRep[i]]] for i in range(len(uniqRep)) ]
    idxs = np.concatenate(np.asarray(idxs))
    return idxs, uniqRep, numRep

def select_action(state, n_actions, chooseRandom = False):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    neigh, neigh_y, neigh_z, labelx, labely, labelz = state
    if (sample > eps_threshold) and (chooseRandom == False):
        with torch.no_grad():
            print('Select action using policy net')
            # print(policy_net(neigh, neigh_y, neigh_z, label).max(1)[1].shape)
            return policy_net(neigh, neigh_y, neigh_z, labelx, labely, labelz).max(1)[1]
    else:
        # print('label size: ', label.shape[0])
        print('Select action using randomization')
        return torch.randint(0, n_actions, (neigh.shape[0],)).cuda()
        # return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


# setting hyperparameters
BATCH_SIZE = 10000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 50
num_episodes = 200

n_actions = 27 # tuple(range(0,10))
patch_height, patch_width = (11,11)
channels = 1

policy_net = DQN(patch_height, patch_width, channels, n_actions).cuda()
target_net = DQN(patch_height, patch_width, channels, n_actions).cuda()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(100000)

steps_done = 0




def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # print(batch.next_state)
    bstate = torch.stack(list(batch.state), dim=0)
    neighbors = bstate[..., 0].cuda()
    neighbors_z = bstate[..., 1].cuda()
    neighbors_y = bstate[..., 2].cuda()
    labels_x = torch.unsqueeze(bstate[..., 3][:,0],1).cuda()
    labels_y = bstate[..., 4].cuda()
    labels_z = bstate[..., 5].cuda()
    # print(neighbors.shape)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([torch.unsqueeze(s,0) for s in batch.next_state
                                                if s is not None])
    # state_batch = torch.cat(batch.state)
    action_batch = torch.vstack(batch.action)
    reward_batch = torch.vstack(batch.reward).cuda()

    state_action_values = policy_net(neighbors, neighbors_z, neighbors_y, labels_x, labels_y, labels_z).gather(1, action_batch)

    nextstatebstate = non_final_next_states
    # print(non_final_next_states.shape)
    next_neighbors = nextstatebstate[..., 0].cuda()
    next_neighbors_z = nextstatebstate[..., 1].cuda()
    next_neighbors_y = nextstatebstate[..., 2].cuda()
    next_labels_x = torch.unsqueeze(nextstatebstate[..., 3][:,0], 1).cuda()
    next_labels_y = nextstatebstate[..., 4].cuda()
    next_labels_z = nextstatebstate[..., 5].cuda()
    # print(nextstatebstate[..., 0].shape)
    next_state_values = torch.zeros(BATCH_SIZE).cuda()

    next_state_values[non_final_mask] = target_net(next_neighbors, next_neighbors_z,
                                                   next_neighbors_y, next_labels_x, next_labels_y, next_labels_z).max(1)[0].detach()

    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss().cuda()
    loss = criterion(state_action_values, expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

