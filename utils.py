"""Helpers and utilities"""

import random
import math
import matplotlib.pyplot as plt
import torch

steps_done = 0  # TODO dirty


def select_action(model, state, n_actions, eps_start=0.9, eps_end=0.05, eps_decay=200):
    """Selects an action using the epsilon-greedy policy. Probability (epsilon) of not using the model decays exponentially.

    Args:
        model: the prediction/policy model to use
        state (batch(1) * channel * height * width tensor): the current state
        n_actions (int): number of possible actions
        eps_start (float, optional): inital value of epsilon. Defaults to 0.9.
        eps_end (float, optional): value of epsilon at infinity. Defaults to 0.05.
        eps_decay (float, optional): rate of decay; bigger is slower. Defaults to 200.

    Returns:
        1 * 1 int tensor: the selected action
    """
    global steps_done
    sample = random.random()
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return model(state).argmax(1).view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)


def reset_steps():
    """Resets the steps_done counter.
    """
    global steps_done
    steps_done = 0


def plot_scores(episode_scores):
    """Plots the rewards of episodes; solved is 200 points. The plot will update after every episode.

    Args:
        epsiode_scores (float list): scores of the different episodes
    """
    plt.figure(2)
    plt.clf()
    scores_t = torch.tensor(episode_scores, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    plt.axhline(200, ls='--')
    plt.draw()
    plt.pause(0.001)  # pause a bit so that plots are updated
