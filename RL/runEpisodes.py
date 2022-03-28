import torch
import numpy as np
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import ConcatDataset
import nibabel as nib
from Dataset import MRDataSet2_render
from torch.autograd import Variable
from itertools import count
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from RL.RL_subcort import take_action, ravel, ravel2d, ReplayMemory, DQN, optimize_model
import random, math

ACTIONS = {
        0:[-1, -1,  0],
        1:[-1,  0,  0],
        2:[-1,  1,  0],
        3:[ 0, -1,  0],
        4:[ 0,  1,  0],
        5:[ 1, -1,  0],
        6:[ 1,  0,  0],
        7:[ 1,  1,  0],
        8:[ 0,  0,  0]
    }
paintbrush = [ [1,0,0],
               [-1,0,0],
                [0,0,0],
               [0,1,0],
               [0,-1,0]]


# TODO: change to make it one slice, full slice
objs = [
        '/data/infant/objects/010_subcortex_z153.obj'
        ]
traindata =[]
pad = 5
width = (2*pad)+1
trainBatchSize = 1
testBatchSize = 1
num_episodes = 100
n_actions = len(ACTIONS)
noaction = len(ACTIONS) - 1
TARGET_UPDATE = 10
# setting hyperparameters
# BATCH_SIZE = 10000
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200


patch_height, patch_width = (11,11)
channels = 1

policy_net = DQN(patch_height, patch_width, channels, n_actions).cuda()
target_net = DQN(patch_height, patch_width, channels, n_actions).cuda()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(100000)

steps_done = 0




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


def get_newlabel_and_calc_reward(label, action, label_vec, gt_label_vec, subsetcoords, subsetDataNum,
                                 vislabel, clusternums, seedStreamCount, paintbrush_acts, gtlabel):
    label_new = label[:]
    # change to equals 2 got rid of all the past history
    label_new[label_new > 0] = 2
    done = np.array([False] * len(action))

    # find coordinates that have already been travelled
    label_ind = np.asarray([label_new[tuple((subsetDataNum, subsetcoords))] > 0]).reshape(-1)
    label_ind = label_ind.astype(np.int)
    # change those seeds to be finished
    # done[ label_ind == 1] = True
    # work with the non-finished seeds
    if type(label_ind) == np.int:
        label_ind = np.asarray([label_ind, ])

    subsetclusternums = clusternums[label_ind == 1]
    paintbrush_coords = subsetcoords + np.expand_dims(paintbrush_acts, 1).transpose()
    # label_new[tuple((subsetDataNum, subsetcoords))] = 1
    label_new[tuple((subsetDataNum, paintbrush_coords))] = 1

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

    # vislabel[tuple((subsetDataNum, subsetcoords))] = clusternums
    vislabel[tuple((subsetDataNum, paintbrush_coords))] = clusternums
    diff = gtlabel[tuple((subsetDataNum, paintbrush_coords))] - label_vec.detach().cpu().numpy()
    nonzeros = np.count_nonzero(diff, 1)
    diff[diff > 0] = 1
    diffavg = np.mean(diff, 1)


    # label_reward = (torch.abs(label_vec - gt_label_vec) * (-1)) + 2  # + (torch.abs(label_vec - gt_label_vec)*(-2))
    label_reward = torch.tensor((-1*nonzeros) + 5)
    # label_reward[label_reward < 2] = 0
    reward = label_reward.type(torch.FloatTensor).cuda() * seedStreamCount
    # try:
    #     if len(label_reward) >1:
    #         done[label_reward.cpu() == 0] = True
    # except TypeError:
    #     if diff == 0:
    #         done = True
    done[diffavg > 0.5] = True

    if action.shape[0] == 1:
        if action == noaction:
            done = True
    else:
        done[action == noaction] = True

    # if torch.all(subsetcoords == subsetcoords[0]):
    #     done[done == False] = True

    return label_new, reward, done, vislabel, seedStreamCount


for i in np.arange(len(objs)):
    tmpdata = MRDataSet2_render.MRDataSet(file=objs[i],
                                          transform=T.Compose([
                                           MRDataSet2_render.ToTensor()
                                       ]), render=True)
    traindata.append(tmpdata)
alldata = ConcatDataset(traindata)

dataloader = DataLoader(alldata, batch_size=trainBatchSize, shuffle=True,
                        num_workers=0, drop_last=False)

dx, dy, dz  = alldata.datasets[0].dataset['cropped_size']
labelBoxSize = dx*dy*dz
''' cropped_indices_map_atlas maps from big area to sample number '''
cropped_indices_map_atlas = np.zeros((len(traindata), labelBoxSize)).astype(np.int)
roi_cropped_indices_map_atlas = np.zeros((len(traindata), labelBoxSize)).astype(np.int)
''' cropped_indices_map_atlas maps from sample number to big area '''
# cropped_indices_map_atlas_nopad = np.zeros((len(traindata), (dx, dy, dz))).astype(np.int)
valids = []
for c in range(len(traindata)):
    cropped_indices_map_atlas[c] = \
        dataloader.dataset.datasets[c].dataset['cropped_indices_map'].ravel()
    roi_cropped_indices_map_atlas[c] = \
        dataloader.dataset.datasets[c].dataset['roi_cropped_indices_map'].ravel()
    valids.append( dataloader.dataset.datasets[c].dataset['cropped_indices'].size)

forindexing = np.zeros(cropped_indices_map_atlas.size)
forindexing[cropped_indices_map_atlas.ravel() > -1] = np.arange(0,len(cropped_indices_map_atlas.ravel()[cropped_indices_map_atlas.ravel() > -1]))
forindexing = forindexing.astype(np.int)

roi_forindexing = np.zeros(roi_cropped_indices_map_atlas.size)
roi_forindexing[roi_cropped_indices_map_atlas.ravel() > -1] = np.arange(0,len(roi_cropped_indices_map_atlas.ravel()[roi_cropped_indices_map_atlas.ravel() > -1]))
roi_forindexing = roi_forindexing.astype(np.int)

tmpindarr = np.expand_dims(np.arange(-1*pad, pad+1),1)
tmpz = (((dy*dz) * np.tile(tmpindarr,width)) + tmpindarr.transpose()).reshape(-1)
tmps = np.empty((3,tmpz.size))
tmps[2] = tmpz
tmps[0] = (((dy*dz) * np.tile(tmpindarr, width)) + (dz * tmpindarr).transpose()).reshape(-1)
tmps[1] = ((dz * tmpindarr) + tmpindarr.transpose()).reshape(-1)

nii = nib.load('/data/infant/T2_train_2021//010-C-T1_T2w.1mm.bse.N4.full+mask_s3.nii.gz')
labelnii = nib.load('/data/infant/t2traindata_labels_handedit_YK_05072020/010-C-T1.T2w.final2.label.nii.gz'
                   ).get_fdata()[59 - pad:123 +pad, 142 -pad:198 + pad, 153-pad:153+pad+1]
gtlabel = labelnii.ravel().reshape(-1, labelnii.size)
data = nii.get_fdata()

train= True
for episode in range(num_episodes):
    print('\n\nEPISODE ', episode+1)
    # master copy of the full size label
    flatlabel = np.zeros_like(cropped_indices_map_atlas)
    vislabel = np.zeros_like(cropped_indices_map_atlas).astype(np.int)
    previous_batch_num = 1
    # plt.ion()
    # fig, ax = plt.subplots()
    # plt.imshow(data[59 - pad:123 +pad, 142 -pad:198 + pad, 153].transpose(), cmap='gray', origin='lower') #
    tmplabel = np.zeros_like(cropped_indices_map_atlas.ravel())
    roi_subsetindex = roi_forindexing[vislabel.ravel() == 0]
    indexTrainData = roi_subsetindex[roi_subsetindex > 0]
    voxelsLeft = len(indexTrainData)

    for idx in count():
        startloc = np.random.choice(indexTrainData,size=trainBatchSize)

        print("BATCH: ", idx)

        if trainBatchSize > 1:
            subsetdata = torch.utils.data.Subset(alldata, startloc.ravel())
            dataloader = DataLoader(subsetdata, batch_size=trainBatchSize, shuffle=True,
                                    num_workers=0, drop_last=False)
            sample, indices_batch, _, cropped_indices, datasetNum = next(iter(dataloader))
            sorted = torch.argsort(cropped_indices)
            cropped_indices = cropped_indices[sorted]
            datasetNum = datasetNum[sorted]
            neighbors = Variable(sample['neighbors'].cuda(), requires_grad=False)[sorted]
            neighbors_z = Variable(sample['neighbors_z'].cuda(), requires_grad=False)[sorted]
            neighbors_y = Variable(sample['neighbors_y'].cuda(), requires_grad=False)[sorted]
            ylabel = Variable(sample['label'].cuda(), requires_grad=False)[sorted]
            actual_batch_num = ylabel.shape[0]

            # Assign each cluster an integer corresponding with the seed number
            clusternums = torch.LongTensor(range(previous_batch_num, actual_batch_num + (previous_batch_num)))
            clusternums = clusternums[sorted]
            seedStreamCount = torch.ones(actual_batch_num).cuda()
        else:
            sample, indices_batch, _, cropped_indices, datasetNum = alldata.__getitem__(int(startloc))
            neighbors = Variable(torch.unsqueeze(sample['neighbors'],0).cuda(), requires_grad=False)
            neighbors_z = Variable(torch.unsqueeze(sample['neighbors_z'],0).cuda(), requires_grad=False)
            neighbors_y = Variable(torch.unsqueeze(sample['neighbors_y'],0).cuda(), requires_grad=False)
            ylabel = Variable(sample['label'].cuda(), requires_grad=False)
            actual_batch_num = trainBatchSize
            cropped_indices = torch.tensor([cropped_indices])

            # Assign each cluster an integer corresponding with the seed number
            clusternums = torch.LongTensor(range(previous_batch_num, actual_batch_num + (previous_batch_num)))
            seedStreamCount = torch.ones(actual_batch_num).cuda()

        # cropped indices is np.where(big area == valid). sample_ids is the numbering for sample subsetting

        # Initialize action choices
        acts = [ravel(value, (dx,dy,dz)) for key, value in ACTIONS.items()]
        acts = torch.tensor(np.repeat(np.expand_dims(np.asarray(acts), axis=0),actual_batch_num, axis=0))
        paintbrush_acts = np.asarray([ravel(paintbrush[i], (dx, dy, dz)) for i in range(0,len(paintbrush))])


        flatlabel[flatlabel > 0] = 2
        # flatlabel[tuple((datasetNum, cropped_indices))] = 1
        paintbrush_coords = cropped_indices + np.expand_dims(paintbrush_acts, 1).transpose()
        flatlabel[tuple((datasetNum, paintbrush_coords))] = 1
        vislabel[tuple((datasetNum, paintbrush_coords))] = clusternums

        DataNumInd = np.repeat(datasetNum, width ** 2, axis=0)
        labels = []
        #TODO: fix boundary issue for flatlabel
        for dim in range(3):
            tmp = np.tile(tmps[dim], (actual_batch_num, 1))
            new_label_coords_xz = (np.expand_dims(cropped_indices.detach().numpy(), 1) + tmp).astype(np.int64).ravel()
            labels.append(torch.tensor(flatlabel[tuple((DataNumInd,
                                                        new_label_coords_xz))].reshape(-1, 1, width,
                                                                                       width).astype(
                'float32')).cuda())

        # Initialize state
        state = (neighbors, neighbors_y, neighbors_z, labels[0], labels[1],labels[2])

        # if n%10 == 0:
        #     axs = Axes3D(fig)
        #     indices1 = np.asarray(np.where(vislabel[0].reshape((dx,dy,dz)) > 0))
        #     ax.scatter(indices1[0], indices1[1], c=vislabel[0][vislabel[0] > 0])
        #     plt.xlim([0,dx])
        #     plt.ylim([0, dy])
        #     axs.axes.set_xlim3d(left=0, right=dx)
        #     axs.axes.set_ylim3d(bottom=0, top=dy)
        #     axs.axes.set_zlim3d(bottom=0, top=dz)
        #     axs.scatter(indices1[0], indices1[1], indices1[2], s=2, c=vislabel[0][vislabel[0] > 0], cmap='prism')
        #     fig.canvas.draw()
        #     plt.pause(0.00001)
        #     plt.show()

        finished = False
        loop = 0
        n = 0
        chooseRandom = False
        while not finished:
            finished = False
            # select action
            actions = select_action(state, n_actions, chooseRandom)
            print('actions taken:', actions)
            chooseRandom = False

            # take the action
            newcoords = take_action(cropped_indices.detach().cpu(), actions.detach().cpu(), acts)
            tmpidx = cropped_indices_map_atlas[tuple((datasetNum,newcoords))]
            try:
                if len(tmpidx) > 0:
                    newcoords[ tmpidx== -1] = -1
            except TypeError:
                newcoords = np.asarray([newcoords])
                datasetNum = np.asarray([datasetNum])
                if tmpidx == -1:
                    break
            non_final_idx = np.where(newcoords != -1)[0]
            subsetcoords = torch.tensor(newcoords[non_final_idx])
            subsetDataNum = datasetNum[non_final_idx]
            clusternums = clusternums[non_final_idx]
            seedStreamCount = seedStreamCount[non_final_idx]
            # print('newcoords: ', cropped_indices_map_atlas[tuple((subsetDataNum,subsetcoords))])
            oneDidxs = ravel2d([subsetDataNum,subsetcoords.detach().numpy()], cropped_indices_map_atlas.shape)
            indexTrainData = forindexing[oneDidxs]
            # print('indices for subset: ', indexTrainData)
            subsetdata = torch.utils.data.Subset(alldata, indexTrainData)
            subsetdataloader = DataLoader(subsetdata, batch_size=actual_batch_num, shuffle=False)

            # get the new coordinates and subset the data and compare
            if len(subsetcoords) >1 :
                ssample, _, _, cropped_indices, _ = next(iter(subsetdataloader))
            elif len(subsetcoords) < 1:
                finished = True
                break
            else:
                ssample, _, _, cropped_indices, _ = alldata.__getitem__(int(indexTrainData))
                cropped_indices = torch.tensor(cropped_indices)
                ssample['neighbors'] = torch.unsqueeze(ssample['neighbors'], 0)
                ssample['neighbors_z']  = torch.unsqueeze(ssample['neighbors_z'], 0)
                ssample['neighbors_y'] = torch.unsqueeze(ssample['neighbors_y'], 0)

            label_vec = ssample['label'].cuda()
            # edges = ssample['edgemap'].cuda()
            try:
                ylabel = ylabel[non_final_idx]
            except IndexError:
                ylabel = ylabel

            flatlabel_1, reward, done, vislabel, seedStreamCount = get_newlabel_and_calc_reward(flatlabel,
                                                                   actions.detach().cpu()[non_final_idx], label_vec,
                                                                    ylabel, subsetcoords, subsetDataNum,
                                                                   vislabel, clusternums, seedStreamCount,
                                                                    paintbrush_acts, gtlabel)
            diff_flatlabel = flatlabel_1 - flatlabel
            rewards = torch.zeros((len(newcoords))).cuda()
            rewards[non_final_idx] = reward
            rewards = rewards.clone().detach()
            print("Reward median, mean, and max: ", torch.median(rewards), torch.mean(rewards), torch.max(rewards))

            # get indices where done is True
            doneTrue = np.where(done == True)[0]
            # unravel current state, make each row its own tuple

            # tmpzeros = torch.zeros((state[0].shape[0],3,width,width)).cuda()
            # tmpstate4 = torch.cat([state[3], tmpzeros], dim=1)
            state_unraveled = torch.cat([torch.unsqueeze(state[0], 4),torch.unsqueeze(state[1], 4),
                               torch.unsqueeze(state[2], 4),torch.unsqueeze(state[3], 4),
                                         torch.unsqueeze(state[4], 4),torch.unsqueeze(state[5], 4)], 4)

            # Observe new state
            '''each in subsetcoords is a one data point in the batch subsetcoords,
            where each point can be from a different data point'''
            subsetDataNumInd = np.repeat(subsetDataNum, width**2, axis=0)

            labels = []
            for dim in range(3):
                tmp = np.tile(tmps[dim], (len(subsetDataNum), 1))
                new_label_coords_xz = (np.expand_dims(subsetcoords, 1) + tmp).astype(np.int64).ravel()
                labels.append(torch.tensor(diff_flatlabel[tuple((subsetDataNumInd,
                                                          new_label_coords_xz))].reshape(-1, 1, width,
                                                                                         width).astype(
                    'float32')).cuda())

            neighbors = ssample['neighbors'].cuda() #[done == False]
            neighbors_z = ssample['neighbors_z'].cuda() #[done == False]
            neighbors_y = ssample['neighbors_y'].cuda() #[done == False]
            next_state = (neighbors, neighbors_y, neighbors_z, labels[0], labels[1], labels[2])

            # nextstate_unraveled = np.concatenate(np.expand_dims(next_state, axis=6), axis=5)
            nextstate_unraveled = [None]*len(newcoords)
            # tmpzeros = torch.zeros((next_state[0].shape[0], 3, width, width)).cuda()
            # tmpstate4 = torch.cat([next_state[3], tmpzeros], dim=1)
            nextstate_unraveled0 = torch.cat([torch.unsqueeze(next_state[0], 4), torch.unsqueeze(next_state[1], 4),
                                         torch.unsqueeze(next_state[2], 4), torch.unsqueeze(next_state[3], 4),
                                              torch.unsqueeze(next_state[4], 4),torch.unsqueeze(next_state[5], 4),], 4)
            # nextstate_unraveled[non_final_idx] = nextstate_unraveled0

            for nf, index in enumerate(non_final_idx):
                nextstate_unraveled[index] = nextstate_unraveled0[nf]
            # nextstate_unraveled = list(tuple(map(tuple, nextstate_unraveled)))
            if len(doneTrue) > 0:
                for t in range(len(doneTrue)):
                    nextstate_unraveled[t] = None

            if np.all(done == True):
                finished = True
                break

            # Store the transition in memory
            print("Number of seeds still active: ", len(state_unraveled), "\n")
            memory.push(state_unraveled, actions, nextstate_unraveled, rewards)

            # Move to the next state
            doneFalse = np.where(done == False)[0]

            state = (neighbors[doneFalse], neighbors_y[doneFalse], neighbors_z[doneFalse], labels[0][doneFalse],
                     labels[1][doneFalse],labels[2][doneFalse])
            datasetNum = datasetNum[doneFalse]

            # cropped_indices = cropped_indices[doneFalse]
            try:
                cropped_indices = cropped_indices[doneFalse]
            except IndexError:
                cropped_indices = cropped_indices

            flatlabel = flatlabel_1[:]
            # prev_indexTrainData = indexTrainData
            # prev_subsetcoords = subsetcoords

            n+=1
            # visualize
            # if n%5 == 0:
            #     # axs = Axes3D(fig)
            #     # indices1 = np.where(vislabel[0].reshape((dx,dy,dz)) > 0)
            #     # axs.axes.set_xlim3d(left=0, right=dx)
            #     # axs.axes.set_ylim3d(bottom=0, top=dy)
            #     # axs.axes.set_zlim3d(bottom=0, top=dz)
            #     # axs.scatter(indices1[0], indices1[1], indices1[2], s=2, c=vislabel[0][vislabel[0] > 0], cmap='prism')
            #     # fig.canvas.draw()
            #     # plt.pause(0.00001)
            #     # plt.show()
            #
                # indices1 = np.asarray(np.where(vislabel[0].reshape((dx, dy, dz)) > 0))
                # ax.scatter(indices1[0], indices1[1], c=vislabel[0][vislabel[0] > 0], cmap='prism')
                # plt.xlim([0, dx])
                # plt.ylim([0, dy])
                # fig.canvas.draw()
                # plt.pause(0.00001)
                # plt.show()
                # print('render plot')

            # Perform one step of the optimization (on the policy network)
            if train:
                optimize_model()

            roi_subsetindex = roi_forindexing[vislabel.ravel() == 0]
            indexTrainData = roi_subsetindex[roi_subsetindex > 0]

            print("\nNumber of voxels left to label: ", voxelsLeft)

            if voxelsLeft == 0:
                finished = True
            if voxelsLeft == len(indexTrainData):
                loop +=1
            if loop > 7:
                chooseRandom = True
            #     finished = True
            # if loop>20:
            #     loop = 0
            #     finished = True
            voxelsLeft = len(indexTrainData)

            # nii = nib.load('/data/infant/T2_train_2021/ANTS_tests/010toAtlas/010-C-T1_T2w.boundbox.z153.mask.nii.gz')
            # for i in range(vislabel.shape[0]):
            #     data = pickle.load(open(objs[i], 'rb'))
            #     origimg = np.zeros(data['origsize'][:3])
            #     xpos, xpos_end = data['bounds'][0]
            #     ypos, ypos_end = data['bounds'][1]
            #     zpos, zpos_end = data['bounds'][2]
            #     # origimg[xpos - (pad + 1):xpos_end + (pad + 2), ypos - (pad + 1):ypos_end + (pad + 2),
            #     # zpos - (pad + 1):zpos_end + (pad + 2)] = \
            #     #     vislabel[i].reshape(data['cropped_size'][:3])
            #     origimg[xpos - (pad + 1):xpos_end + (pad + 1), ypos - (pad + 1):ypos_end + (pad + 1),
            #     zpos - (pad + 1):zpos_end + (pad + 1)] = \
            #         vislabel[i].reshape((dx,dy,dz))
            #     recon = nib.Nifti1Image(origimg.astype(np.int16), affine=nii._affine)
            #     nib.save(recon, objs[i].split('.')[0] + '153_' + str(n) + '.nii.gz')

            print('n:', n)

            # test = runRL_inference(objs, tmps, forindexing, (dx, dy, dz), cropped_indices_map_atlas, alldata,
            #                        episode, testBatchSize, width)
            # test.RL_inference(policy_net)

        previous_batch_num = previous_batch_num+actual_batch_num

        # Update the target network, copying all weights and biases in DQN
        if n % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        roi_subsetindex = roi_forindexing[vislabel.ravel() == 0]
        indexTrainData = roi_subsetindex[roi_subsetindex > 0]
        # TODO: figure out why the termination criteria isn't working
        if len(indexTrainData) < 2:
            break
        print("\nNumber of voxels left to label: ", len(indexTrainData))
        subsetdata = torch.utils.data.Subset(alldata, indexTrainData)
        dataloader = DataLoader(subsetdata, batch_size=trainBatchSize, shuffle=True)

    cep = (episode + 1)


    if (episode + 1) % 5 == 0:
        labelnii = np.zeros_like(data)
        labelnii[64 - pad:115 + pad, 142 - pad:190 + pad, 153-pad:153+pad+1] = vislabel[0].reshape(dx,dy,dz)
        recon = nib.Nifti1Image(labelnii, nii._affine)
        nib.save(recon, '/data/infant/objects/010.z153.EP{0}.nii.gz'.format(episode))
        checkpoints_folder = '/data/infant/checkpoints/'
        modelname = '/{0}/RL_subcort_{1}.pth'.format(checkpoints_folder, episode)
        torch.save(policy_net.state_dict(), modelname)
        train = True
    #
    #     test = runRL_inference(objs, tmps, forindexing, (dx, dy, dz), cropped_indices_map_atlas, alldata,
    #                     episode, 10000, width)
    #     test.RL_inference(policy_net)

        # evaluation


# # plt.ioff()
