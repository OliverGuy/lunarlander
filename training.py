import torch
import torch.nn.functional as F


def optimize_model(
        device,
        pred_net,
        target_net,
        optimizer,
        memory,
        loss_func=F.smooth_l1_loss,
        batch_size=128,
        gamma=0.999):
    """Performs a single optimisation step.
    Samples a batch, concatenates the tensors and computes the loss.

    Args:
        device: the torch device in use
        pred_net: the "fast-learning" prediction/policy neural network
        target_net: the "slow-learning" target neural network
        optimizer: the optimizer to use for back propagation
        memory (ReplayMemory): see `memory.py`
        loss_func (func, optional): the `nn.functional` loss function to be used for back propagation. Defaults to `F.smooth_l1_loss`.
        batch_size (int, optional): the batch size to train on. Calling this function does nothing if len(memory) < batch_size. Defaults to 128.
        gamma (float, optional): the discount factor for future rewards. Defaults to 0.999.
    """

    if len(memory) < batch_size:
        return
    batch = memory.batch(batch_size)

    # Compute a mask of non-final states and concatenate the batch elements
    # final states are characterised by next_state = None in the training
    non_final_mask = torch.tensor(tuple(s is not None for s in batch.next_state),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = pred_net(state_batch).gather(dim=1, index=action_batch)

    # Compute max_a Q(s_{t+1}, a) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = reward_batch.to(torch.float) + (gamma * next_state_values)
    # Compute Huber loss
    loss = loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in pred_net.parameters():
        param.grad.data.clamp_(-1, 1)  # clip the gradient
    optimizer.step()
    return loss.item()
