import torch
import numpy as np
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
import pickle
import nibabel as nib
## load data
from torch.autograd import Variable
from itertools import count

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
paintbrush = [ [1,0,0],
               [-1,0,0],
               [0,0,1],
               [0,0,-1]]

ravel = lambda x, y: (y[2] * y[1] * x[0]) + (y[2] * x[1]) + x[2]
ravel2d = lambda x, y: (y[1] * x[0]) + x[1]


subj = '010'
image_dir = '/data/infant/T2_train_2021/'
image_file_suffix = '-C-T1_T2w.1mm.bse.N4.full+mask_s3.nii.gz'

class runRL_inference(object):
    def __init__(self, objs, tmps, forindexing, dims,cropped_indices_map_atlas, alldata,
                 episode, BATCH_SIZE, width=11):
        self.objs = objs
        self.tmps = tmps
        self.forindexing = forindexing
        self.dx, self.dy, self.dz = dims
        self.cropped_indices_map_atlas = cropped_indices_map_atlas
        self.alldata = alldata
        self.BATCH_SIZE = BATCH_SIZE
        self.width = width
        self.episode = episode
        self.paintbrush = np.asarray([ravel(paintbrush[i], dims) for i in range(len(paintbrush))])

    def select_action_eval(self, state, policy_net):
        neigh, neigh_y, neigh_z, labelx, labely, labelz = state
        with torch.no_grad():
            print('Select action using policy net')
            return policy_net(neigh, neigh_y, neigh_z, labelx, labely, labelz).max(1)[1]

    def take_action(self,coord, action, acts):
        newcoord = coord + acts[tuple((np.arange(0, len(action)), action))]
        # print('newcoord shape:' , newcoord.shape )
        # newcoord[action == 9] = -1
        # newcoord[coord == -1] = -1
        return newcoord


    def get_newlabel(self,label, action, subsetcoords, subsetDataNum,
                                     vislabel, clusternums, seedStreamCount):
        label_new = label[:]
        # change to equals 2 got rid of all the past history
        label_new[ label_new > 0] = 2
        done=np.array([False]*len(action))

        # find coordinates that have already been travelled
        label_ind = np.asarray([label_new[tuple((subsetDataNum, subsetcoords))] > 0]).reshape(-1)
        label_ind = label_ind.astype(np.int)
        # change those seeds to be finished
        # done[ label_ind == 1] = True
        # work with the non-finished seeds
        if type(label_ind) == np.int:
            label_ind = np.asarray([label_ind,])

        subsetclusternums = clusternums[label_ind == 1]
        paintbrush_coords = subsetcoords + np.expand_dims(self.paintbrush,1).transpose()
        label_new[tuple((subsetDataNum,subsetcoords))] = 1
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

        vislabel[tuple((subsetDataNum,subsetcoords))] = clusternums
        vislabel[tuple((subsetDataNum, paintbrush_coords))] = clusternums

        if action.shape[0] == 1:
            if action == 27:
                done = True
        else:
            done[action == 27] = True

        # if torch.all(subsetcoords == subsetcoords[0]):
        #     done[done == False] = True

        return label_new, done, vislabel

    def RL_inference(self, policy_net):

        flatlabel = np.zeros_like(self.cropped_indices_map_atlas)
        vislabel = np.zeros_like(self.cropped_indices_map_atlas).astype(np.int)
        previous_batch_num = 1
        pad = 5
        # fig = plt.figure()
        tmplabel = np.zeros_like(self.cropped_indices_map_atlas.ravel())
        dataloader = DataLoader(self.alldata, batch_size=self.BATCH_SIZE, shuffle=True,
                                num_workers=2, drop_last=False)

        subsetindex = self.forindexing[vislabel.ravel() == 0]
        indexTrainData = subsetindex[subsetindex > 0]
        voxelsLeft = len(indexTrainData)

        for idx in count():
            print("BATCH: ", idx)
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
            clusternums = torch.LongTensor(range(previous_batch_num,actual_batch_num + (previous_batch_num)))
            clusternums = clusternums[sorted]
            seedStreamCount = torch.ones(actual_batch_num).cuda()

            # Initialize action choices
            acts = [ravel(value, (self.dx,self.dy,self.dz)) for key, value in ACTIONS.items()]
            acts = torch.tensor(np.repeat(np.expand_dims(np.asarray(acts), axis=0),actual_batch_num, axis=0))

            flatlabel[flatlabel > 0] = 2
            flatlabel[tuple((datasetNum, cropped_indices))] = 1
            vislabel[tuple((datasetNum, cropped_indices))] = clusternums
            paintbrush_coords = cropped_indices + np.expand_dims(self.paintbrush, 1).transpose()
            vislabel[tuple((datasetNum, paintbrush_coords))] = clusternums

            DataNumInd = np.repeat(datasetNum, self.width ** 2, axis=0)
            labels = []
            for dim in range(3):
                tmp = np.tile(self.tmps[dim], (actual_batch_num, 1))
                new_label_coords_xz = (np.expand_dims(cropped_indices.detach().numpy(), 1) + tmp).astype(np.int64).ravel()
                labels.append(torch.tensor(flatlabel[tuple((DataNumInd,
                                                            new_label_coords_xz))].reshape(-1, 1, self.width,
                                                                                           self.width).astype(
                    'float32')).cuda())

            # Initialize state
            state = (neighbors, neighbors_y, neighbors_z, labels[0], labels[1],labels[2])
            finished = False

            loop = 0
            while finished == False:
                # select action
                actions = self.select_action_eval(state, policy_net)
                print('actions taken:', actions)

                # take the action
                newcoords = self.take_action(cropped_indices.detach().cpu(), actions.detach().cpu(), acts)
                tmpidx = self.cropped_indices_map_atlas[tuple((datasetNum,newcoords))]
                try:
                    if type(tmpidx) == np.int64:
                        if tmpidx == -1:
                            newcoords = -1
                            break
                    else:
                        newcoords[ tmpidx== -1] = -1
                except IndexError:
                    if newcoords == -1:
                        break
                non_final_idx = np.where(newcoords != -1)[0]

                subsetcoords = newcoords[non_final_idx]
                subsetDataNum = datasetNum[non_final_idx]
                clusternums = clusternums[non_final_idx]
                seedStreamCount = seedStreamCount[non_final_idx]
                # print('newcoords: ', cropped_indices_map_atlas[tuple((subsetDataNum,subsetcoords))])
                oneDidxs = ravel2d([subsetDataNum,subsetcoords], self.cropped_indices_map_atlas.shape)
                indexTrainData = self.forindexing[oneDidxs]
                # print('indices for subset: ', indexTrainData)
                subsetdata = torch.utils.data.Subset(self.alldata, indexTrainData)
                subsetdataloader = DataLoader(subsetdata, batch_size=actual_batch_num, shuffle=False)

                # get the new coordinates and subset the data and compare
                if (len(subsetcoords) >1) or (len(subsetcoords)==1):
                    if len(subsetcoords) >1 :
                        ssample, _, _, cropped_indices, _ = next(iter(subsetdataloader))
                    elif len(subsetcoords) == 1:
                        ssample, _, _, cropped_indices, _ = self.alldata.__getitem__(int(indexTrainData))
                        cropped_indices = torch.tensor(cropped_indices)
                        ssample['neighbors'] = torch.unsqueeze(ssample['neighbors'], 0)
                        ssample['neighbors_z']  = torch.unsqueeze(ssample['neighbors_z'], 0)
                        ssample['neighbors_y'] = torch.unsqueeze(ssample['neighbors_y'], 0)

                    ylabel = ylabel[non_final_idx]

                    flatlabel, done, vislabel  = self.get_newlabel(flatlabel, actions.detach().cpu()[non_final_idx],
                                                                   subsetcoords, subsetDataNum,
                                                                    vislabel, clusternums, seedStreamCount)
                    print('seed number:', clusternums)

                    # get indices where done is True
                    doneTrue = np.where(done == True)[0]

                    state_unraveled = torch.cat([torch.unsqueeze(state[0], 4),torch.unsqueeze(state[1], 4),
                                       torch.unsqueeze(state[2], 4),torch.unsqueeze(state[3], 4),
                                                 torch.unsqueeze(state[4], 4),torch.unsqueeze(state[5], 4)], 4)

                    # Observe new state
                    '''each in subsetcoords is a one data point in the batch subsetcoords,
                    where each point can be from a different data point'''
                    subsetDataNumInd = np.repeat(subsetDataNum, self.width**2, axis=0)

                    labels = []
                    for dim in range(3):
                        tmp = np.tile(self.tmps[dim], (len(subsetDataNum), 1))
                        new_label_coords_xz = (np.expand_dims(subsetcoords.detach().numpy(), 1) + tmp).astype(np.int64).ravel()
                        labels.append(torch.tensor(flatlabel[tuple((subsetDataNumInd,
                                                                  new_label_coords_xz))].reshape(-1, 1, self.width,
                                                                                                 self.width).astype(
                            'float32')).cuda())

                    neighbors = ssample['neighbors'].cuda() #[done == False]
                    neighbors_z = ssample['neighbors_z'].cuda() #[done == False]
                    neighbors_y = ssample['neighbors_y'].cuda() #[done == False]
                    next_state = (neighbors, neighbors_y, neighbors_z, labels[0], labels[1], labels[2])

                    # nextstate_unraveled = np.concatenate(np.expand_dims(next_state, axis=6), axis=5)
                    nextstate_unraveled = [None]*len(newcoords)

                    nextstate_unraveled0 = torch.cat([torch.unsqueeze(next_state[0], 4), torch.unsqueeze(next_state[1], 4),
                                                 torch.unsqueeze(next_state[2], 4), torch.unsqueeze(next_state[3], 4),
                                                      torch.unsqueeze(next_state[4], 4),torch.unsqueeze(next_state[5], 4),], 4)
                    for nf, index in enumerate(non_final_idx):
                        nextstate_unraveled[index] = nextstate_unraveled0[nf]
                    # nextstate_unraveled = list(tuple(map(tuple, nextstate_unraveled)))
                    if len(doneTrue) > 0:
                        for t in range(len(doneTrue)):
                            nextstate_unraveled[t] = None

                    if np.all(done == True):
                        finished = True

                    # Store the transition in memory
                    print("Number of seeds still active: ", len(state_unraveled), "\n")


                    # Move to the next state
                    doneFalse = np.where(done == False)[0]

                    state = (neighbors[doneFalse], neighbors_y[doneFalse], neighbors_z[doneFalse], labels[0][doneFalse],
                             labels[1][doneFalse],labels[2][doneFalse])
                    datasetNum = datasetNum[doneFalse]

                    try:
                        cropped_indices = cropped_indices[doneFalse]
                    except IndexError:
                        cropped_indices = cropped_indices

                    subsetindex = self.forindexing[vislabel.ravel() == 0]
                    indexTrainData = subsetindex[subsetindex > 0]

                    print("\nNumber of voxels left to label: ", voxelsLeft)
                    if voxelsLeft == len(indexTrainData):
                        loop += 1
                    if loop > 10:
                        finished = True
                    voxelsLeft = len(indexTrainData)

                elif len(subsetcoords) < 1:
                    finished = True

            if loop < 10:
                previous_batch_num = actual_batch_num + previous_batch_num

            subsetindex = self.forindexing[vislabel.ravel() == 0]
            indexTrainData = subsetindex[subsetindex > 0]
            # TODO: figure out why the termination criteria isn't working
            if len(indexTrainData) < 2:
                break
            print("\nOuter loop: Number of voxels left to label: ", len(indexTrainData))
            subsetdata = torch.utils.data.Subset(self.alldata, indexTrainData)
            dataloader = DataLoader(subsetdata, batch_size=self.BATCH_SIZE, shuffle=True)

            # evaluation

        nii = nib.load('{0}/{1}{2}'.format(image_dir, subj, image_file_suffix))
        for i in range(vislabel.shape[0]):
            data = pickle.load(open(self.objs[i], 'rb'))
            origimg = np.zeros(data['origsize'][:3])
            xpos, xpos_end = data['bounds'][0]
            ypos, ypos_end = data['bounds'][1]
            zpos, zpos_end = data['bounds'][2]
            origimg[xpos - (pad + 1):xpos_end + (pad + 2), ypos - (pad + 1):ypos_end + (pad + 2),
            zpos - (pad + 1):zpos_end + (pad + 2)] = \
                vislabel[i].reshape(data['cropped_size'][:3])
            recon = nib.Nifti1Image(origimg.astype(np.int16), affine=nii._affine) # .astype(np.int16)
            nib.save(recon, self.objs[i].split('.')[0] + '_eval_' + str(self.episode) + '.nii.gz')


# load in trained model and run inference

# from RL_subcort import DQN
# checkpoints_folder = '/data/infant/checkpoints/'
# episode = 199
# modelname = '/{0}/RL_subcort_{1}.pth'.format(checkpoints_folder, episode)
# patch_height, patch_width = (11,11)
# channels = 1
# n_actions = 27
# policy_net = DQN(patch_height, patch_width, channels, n_actions).cuda()
# policy_net.load_state_dict(torch.load(modelname))
#
#
# objs = [
#         '/data/infant/objects/010_subcortex.obj',
#         '/data/infant/objects/023_subcortex.obj'
#         ] # '/data/infant/objects/132_subcortex.obj',
#         #'/data/infant/objects/087_subcortex.obj'
# traindata =[]
# pad = 5
# width = (2*pad)+1
#
# for i in np.arange(len(objs)):
#     tmpdata = MRDataSet2_render.MRDataSet(file=objs[i],
#                                        transform=T.Compose([
#                                            MRDataSet2_render.ToTensor()
#                                        ]),render=True)
#     traindata.append(tmpdata)
# alldata = ConcatDataset(traindata)
#
#
# dx, dy, dz  = alldata.datasets[0].dataset['cropped_size']
# labelBoxSize = (dx*dy*dz)
# cropped_indices_map_atlas = np.zeros((len(traindata), labelBoxSize)).astype(np.int)
# for c in range(len(traindata)):
#     cropped_indices_map_atlas[c] = alldata.datasets[c].dataset['cropped_indices_map']
# forindexing = np.zeros(cropped_indices_map_atlas.size)
# forindexing[cropped_indices_map_atlas.ravel() > -1] = np.arange(0,len(cropped_indices_map_atlas.ravel()[cropped_indices_map_atlas.ravel() > -1]))
# forindexing = forindexing.astype(np.int)
# tmpindarr = np.expand_dims(np.arange(-1*pad, pad+1),1)
# tmpz = (((width**2) * np.tile(tmpindarr,width)) + tmpindarr.transpose()).reshape(-1)
# tmps = np.empty((3,tmpz.size))
# tmps[2] = tmpz
# tmps[0] = (((width ** 2) * np.tile(tmpindarr, width)) + (width * tmpindarr).transpose()).reshape(-1)
# tmps[1] = ((width * tmpindarr) + tmpindarr.transpose()).reshape(-1)
# test = runRL_inference(objs, tmps, forindexing, (dx, dy, dz), cropped_indices_map_atlas, alldata,
#                         episode, 1, width)
# test.RL_inference(policy_net)

for i in range(0,10):
    print('outer loop', i)
    if (i!=0) & (i%2 == 0):
        print('repeat',i)
        i -= 1
        print('minus 1', i)