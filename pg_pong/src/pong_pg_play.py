# Parse arguments
import argparse
parser = argparse.ArgumentParser(
    description='PyTorch stochastic policy gradient example on Pong. (PLAY)')
parser.add_argument('--interval', type=float, default=2e-2, metavar='F',
                    help='interval between frames (default: 2e-2)')
parser.add_argument('--seed', type=int, default=30820172044, metavar='N',
                    help='random seed (default: 30820172044)')
parser.add_argument('--plot_env', action='store_true', default=False,
                    help='plot preprocessed environment')
parser.add_argument('--force_cpu', action='store_true', default=False,
                    help='force use of cpu for computation')
parser.add_argument('load_model', type=str, metavar='<path>',
                    help='indicates model to load')
HPARAMS = parser.parse_args()

import gym
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# If CUDA is available (and not force cpu param set) use GPU
use_cuda = torch.cuda.is_available() and not HPARAMS.force_cpu
if use_cuda:
    print("\n\t!!! CUDA is available. Using GPU !!!\n")

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


class PolicyGradient(nn.Module):
    """
    It's out model class.
    """

    def __init__(self, in_dim):
        super(PolicyGradient, self).__init__()
        self.hidden = nn.Linear(in_dim, 200)
        self.out = nn.Linear(200, 3)

        self.rewards = []
        self.actions = []

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 'n' is number of inputs to each neuron
                n = len(m.weight.data[1])
                # "Xavier" initialization
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        h = F.relu(self.hidden(x))
        logits = self.out(h)
        return F.softmax(logits)

    def reset(self):
        del self.rewards[:]
        del self.actions[:]


def preprocess(img):
    """ Preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    if HPARAMS.plot_env:
        plt.imshow(img)
        plt.title("Before")
        plt.show()

    I = img[35:195]     # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0     # erase background (background type 1)
    I[I == 109] = 0     # erase background (background type 2)
    I[I != 0] = 1       # everything else (paddles, ball) just set to 1

    if HPARAMS.plot_env:
        plt.imshow(I, cmap='gray')
        plt.title("After")
        plt.show()

    return I.astype(np.float).ravel()


def get_action(policy, observation):
    # Get current state, which is difference between current and previous state
    cur_state = preprocess(observation)
    state = cur_state - get_action.prev_state \
        if get_action.prev_state is not None else np.zeros(len(cur_state))
    get_action.prev_state = cur_state

    var_state = Variable(
        # Make torch FloatTensor from numpy array and add batch dimension
        torch.from_numpy(state).type(FloatTensor).unsqueeze(0)
    )
    probabilities = policy(var_state)
    # Stochastic policy: roll a biased dice to get an action
    action = probabilities.multinomial()
    # Record action for future training
    policy.actions.append(action)
    # '+ 1' converts action to valid Pong env action
    return action.data[0, 0] + 1


# Used in computing the difference frame
get_action.prev_state = None

###############################
### Whole logic lives below ###
###############################

# Initialize torch with seed
torch.manual_seed(HPARAMS.seed)

# Create environment
env = gym.make('Pong-v0').unwrapped
env.seed(HPARAMS.seed)

# Prepare model, optimizer, environment etc.
running_reward = None
num_episodes = 0
reward_sum = 0
in_dim = 80 * 80  # input dimensionality: 80x80 grid
done = False

policy = PolicyGradient(in_dim)
if HPARAMS.load_model != '':
    policy.load_state_dict(torch.load(HPARAMS.load_model))
    num_episodes = int(HPARAMS.load_model.split('_')[-1].split('.')[0])

if use_cuda:
    policy.cuda()

observation = env.reset()

# Let's play the game ;)
while not done:
    env.render()
    time.sleep(HPARAMS.interval)

    ### Here actions are taken in environment ###
    action = get_action(policy, observation)
    observation, reward, done, _ = env.step(action)
    # Record reward for future training
    policy.rewards.append(reward)
