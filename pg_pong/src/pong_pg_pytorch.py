#######################################################################
### This work is based on: http://karpathy.github.io/2016/05/31/rl/ ###
### Thank you Karpathy for all your work!                           ###
### Also great thanks for all people creating PyTorch. Great work!  ###
#######################################################################

# Parse arguments
import argparse
parser = argparse.ArgumentParser(
    description='PyTorch stochastic policy gradient example on Pong.')
parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gamma', '-g', type=float, default=0.99, metavar='G',
                    help='reward discount factor (default: 0.99)')
parser.add_argument('--seed', '-s', type=int, default=30820172044, metavar='N',
                    help='random seed (default: 30820172044)')
parser.add_argument('--batch_size', '-bs', type=int, default=10, metavar='N',
                    help='update after N games (default: 10)')
parser.add_argument('--render', '-r', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--plot_env', '-pe', action='store_true', default=False,
                    help='plot preprocessed environment')
parser.add_argument('--force_cpu', '-fc', action='store_true', default=False,
                    help='force use of cpu for computation')
parser.add_argument('--save_each', '-se', type=int, default=100, metavar='N',
                    help='save model each N episodes (default: 100)')
parser.add_argument('--save_dir', '-sd', type=str, default='.', metavar='<dir>',
                    help='indicates directory where to save model \
                    (default: current directory)')
parser.add_argument('--load_model', '-lm', type=str, default='', metavar='<path>',
                    help='indicates model to load and continue learning \
                    (default: start with "Xavier" initialization)')
HPARAMS = parser.parse_args()

import gym
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
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

policy = PolicyGradient(in_dim)
if HPARAMS.load_model != '':
    policy.load_state_dict(torch.load(HPARAMS.load_model))
    num_episodes = int(HPARAMS.load_model.split('_')[-1].split('.')[0])

if use_cuda:
    policy.cuda()

optimizer = optim.RMSprop(
    policy.parameters(),
    lr=HPARAMS.learning_rate,
    weight_decay=HPARAMS.weight_decay
)
optimizer.zero_grad()

observation = env.reset()

# Let's play the game ;)
while True:
    if HPARAMS.render:
        env.render()

    ### Here actions are taken in environment ###
    action = get_action(policy, observation)
    observation, reward, done, _ = env.step(action)
    # Record reward for future training
    policy.rewards.append(reward)
    reward_sum += reward

    ### Here is our reinforcement learning logic ###
    if done:
        num_episodes += 1

        # Compute discounted reward
        discounted_R = []
        running_add = 0
        for reward in policy.rewards[::-1]:
            if reward != 0:
                # Reset the sum, since this was a game boundary (pong specific!)
                running_add = 0

            running_add = running_add * HPARAMS.gamma + reward
            # "Further" actions have less discounted rewards
            discounted_R.insert(0, running_add)

        rewards = FloatTensor(discounted_R)
        # Standardize rewards
        rewards = (rewards - rewards.mean()) / \
            (rewards.std() + np.finfo(np.float32).eps)
        # Batch size shouldn't influence update step
        rewards = rewards / HPARAMS.batch_size

        # Reinforce actions
        for action, reward in zip(policy.actions, rewards):
            action.reinforce(reward)

        # BACKPROP!!! (Gradients are accumulated each episode until update)
        autograd.backward(policy.actions, [None for a in policy.actions])

        ### Here we do weight update each batch ###
        if num_episodes % HPARAMS.batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            print "### Updated parameters! ###"

        ### Here we reset "learning state" and do book-keeping ###
        # Resetting
        policy.reset()
        observation = env.reset()  # reset env
        get_action.prev_state = None

        # Book-keeping
        running_reward = reward_sum if running_reward is None else \
            running_reward * 0.99 + reward_sum * 0.01
        print '{:>5} | {} | Episode reward total was {:d}. Running mean: {:.5f}' \
            .format(num_episodes, datetime.now().strftime('%H:%M:%S'),
                    int(reward_sum), running_reward)
        if num_episodes % HPARAMS.save_each == 0:
            directory = HPARAMS.save_dir
            if len(directory) > 0 and directory[-1] == '/':
                directory = directory[0:-1]

            path = "{}/model_rr_{:.3f}_epi_{}.pt".format(
                directory, running_reward, num_episodes)
            torch.save(policy.state_dict(), path)
            print "### Saved model: {} ###".format(path)

        reward_sum = 0
