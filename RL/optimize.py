import torch.nn as nn
import torch

def optimize_model(memory, batch_size, Transition, gamma, optimizer, policy_net, target_net):
    if len(memory) < batch_size +1 :
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    bimages = torch.stack(list(batch.image), dim=0).view(-1,1,11, 11)
    blabels = torch.stack(list(batch.label), dim=0).view(-1,3,11, 11)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_image)), dtype=torch.bool)
    non_final_next_images = torch.cat([torch.unsqueeze(s,0) for s in batch.next_image
                                                if s is not None]).view(-1,1,11, 11)
    non_final_next_labels = torch.cat([torch.unsqueeze(s, 0) for s in batch.next_label
                                       if s is not None]).view(-1,3,11, 11)
    # state_batch = torch.cat(batch.state)
    action_batch = torch.vstack(batch.action)
    reward_batch = torch.vstack(batch.reward).cuda()

    state_action_values = policy_net(bimages.cuda(), blabels.cuda()).gather(1, action_batch)

    next_state_values = torch.zeros(batch_size).cuda()

    next_state_values[non_final_mask] = target_net(non_final_next_images, non_final_next_labels).max(1)[0].detach()

    expected_state_action_values = (next_state_values.unsqueeze(1) * gamma) + reward_batch

    # do this twice
    criterion = nn.SmoothL1Loss().cuda()
    loss = criterion(state_action_values, expected_state_action_values)

    # add losses

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()