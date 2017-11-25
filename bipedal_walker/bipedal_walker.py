import gym
from itertools import count
from collections import namedtuple
import torch, torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

NUM_EPISODES = 50
SCREEN_WIDTH = 400
SCREEN_LENGTH =600

WINDOW_MAX_Y = 300
WINDOW_MIN_Y = 200
TRANSITION_BUFFER_SIZE = 100

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.input = nn.Linear(24, 60)
        self.hidden1 = nn.Linear(60, 20)
        self.hidden2 = nn.Linear(20, 10)
        self.output = nn.Linear(10, 4)

    def forward(self,x):
        x = F.tanh(self.input(x))
        x = F.tanh(self.hidden1(x))
        x = F.tanh(self.hidden2(x))
        return F.tanh(self.output(x))


class TransitionBuffer:
    def __init__(self):
        self.buffer = []

    def store_transition(self,transition):
        self.buffer.append(transition)
        if len(self.buffer) > TRANSITION_BUFFER_SIZE:
            self.buffer.pop(0)

    def get_transitions(self):
        return self.buffer


def update_model():
    pass


def get_state(obs):
    return Variable(torch.from_numpy(obs).float(), volatile=True)


if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2').unwrapped
    env.reset()
    model = DQN()
    transition_buffer = TransitionBuffer()
    Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))
    for episode in range(NUM_EPISODES):
        env.reset()
        action = env.action_space.sample()

        previous_state = get_state(np.zeros(24))
        current_state = get_state(np.zeros(24))

        for i in count():
            env.render(mode='rgb_array')
            action = model(current_state).data

            obs, reward, done, info = env.step(action)

            #Updating states
            previous_state = current_state
            current_state = get_state(obs)

            #Storing transition
            transition_buffer.store_transition(Transition(previous_state,action,reward,current_state))

            #Updating model
            update_model()

            if done is True:
                break