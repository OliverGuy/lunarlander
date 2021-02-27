import gym
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F

from memory import ReplayMemory
from DQN import DQN
from preprocessing import get_screen
from utils import select_action, plot_scores
from training import optimize_model

# based on https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html


# XXX unwrapped ? cf
# https://discuss.pytorch.org/t/in-the-official-q-learning-example-what-does-the-env-unwrapped-do-exactly/28695
env = gym.make('LunarLander-v2')

plt.ion()  # interactive mode, to draw in non-blocking mode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x80x180
# which is the result of a down-scaled render buffer in get_screen()
init_screen = get_screen(env)
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
n_actions = env.action_space.n

# fast-learning prediction/policy net
pred_net = DQN(screen_height, screen_width, n_actions).to(device)
# slow-learning target network
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(pred_net.state_dict())
target_net.eval()  # freeze the weights of the target net


# training loop parameters

REPLAY_MEMORY_SIZE = 10000
memory = ReplayMemory(REPLAY_MEMORY_SIZE)

TARGET_UPDATE = 10000  # period of target network update
optimizer = optim.RMSprop(pred_net.parameters())

# TODO make weights update be after a certain step count

### TRAINING LOOP ###
num_episodes = 50
episode_rewards = []
steps = 0
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen(env).to(device)
    current_screen = get_screen(env).to(device)
    state = current_screen - last_screen
    episode_rewards.append(0)
    done = False
    while not done:
        # Select and perform an action
        action = select_action(pred_net, state, n_actions).to(device)
        _, reward, done, _ = env.step(action.item())  # our states are screenshot differences
        episode_rewards[-1] += reward

        reward = torch.tensor([reward], device=device)

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen(env).to(device)
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model(device, pred_net, target_net, optimizer, memory)
        steps += 1

        if steps == TARGET_UPDATE:  # update the target net weights
            steps = 0
            target_net.load_state_dict(pred_net.state_dict())
        
    plot_scores(episode_rewards)

print('Done')
env.render()
env.close()
plt.ioff()
plt.show()
