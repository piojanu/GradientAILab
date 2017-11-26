import gym
from itertools import count
from collections import namedtuple
import torch, torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random

ByteTensor = torch.ByteTensor

NUM_EPISODES = 500
SCREEN_WIDTH = 400
SCREEN_LENGTH =600
WINDOW_MAX_Y = 300
WINDOW_MIN_Y = 200
BATCH_SIZE = 100
GAMMA = 0.9
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.input = nn.Linear(24, 72)
        self.hidden = nn.Linear(72, 36)
        self.output = nn.Linear(36, 16)

    def forward(self,x):
        x = F.tanh(self.input(x))
        x = F.tanh(self.hidden(x))
        return F.tanh(self.output(x))


class TransitionBuffer:
    def __init__(self):
        self.buffer = []
        for i in range(BATCH_SIZE):
            self.buffer.append([get_state(np.zeros(24)),torch.FloatTensor([0]),torch.FloatTensor([0]),get_state(np.zeros(24))])

    def push(self,transition):
        self.buffer.append(transition)
        if len(self.buffer) > BATCH_SIZE:
            self.buffer.pop(0)

    def get_batch(self):
        #Transposing list of lists
        return list(zip(*self.buffer))


def update_model(model, transition_buffer,optimizer):
    # Compute a mask of non-final states and concatenate the batch elements
    states,actions,rewards,next_states = transition_buffer.get_batch()

    non_final_mask = [i for i,state in enumerate(next_states) if state is not None]
    non_final_mask = torch.LongTensor(non_final_mask)
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in next_states if s is not None]).view(-1,24), volatile=True)

    #print(non_final_next_states)

    states = Variable(torch.cat(list(states)).view(-1,24))
    actions = Variable(torch.cat(list(actions)).view(-1,1).type(torch.LongTensor))
    rewards = Variable(torch.cat(list(rewards)))

    #print("States:",states,", actions:",actions,", rewards:",rewards)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    q_values = model(states).gather(1, actions)
    #print("Qvals:",q_values)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(torch.Tensor))

    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]

    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + rewards

    # Compute Huber loss
    loss = F.smooth_l1_loss(q_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def get_state(obs):
    return torch.from_numpy(obs).float()

#Returns the index of the action with max value according to our DQN model
def get_action(model,state):
    #chance = random.random()
    #if chance >
    return int(model(state).data.max(0)[1].numpy()[0])

def get_action_vec(action_ind):
    #Getting binary vector of action ie. 9 is 1001
    action_vec = np.array([int(bit) for bit in '{0:04b}'.format(action_ind)])

    #Changing value scope from {0,1} to {-1,1}
    return action_vec*2 - 1

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2').unwrapped
    env.reset()
    model = DQN()
    transition_buffer = TransitionBuffer()

    optimizer = optim.RMSprop(model.parameters())

    for episode in range(NUM_EPISODES):
        env.reset()
        action_vec = env.action_space.sample()

        previous_state = get_state(np.zeros(24))
        current_state = get_state(np.zeros(24))

        for i in count():
            #Rendering screen
            env.render(mode='rgb_array')

            #Getting best action index
            action_ind = get_action(model,Variable(current_state, volatile=True))

            #Getting action for each joint, 4 values {-1,1}
            action_vec = get_action_vec(action_ind)

            #print("action_id =",action_ind,", action vec =",action_vec)

            #Executing step according to our action
            obs, reward, done, info = env.step(action_vec)

            #Updating states
            previous_state = current_state
            if done is False:
                current_state = get_state(obs)
            else:
                current_state = None

            #Storing transition
            transition_buffer.push([previous_state,torch.FloatTensor([action_ind]),torch.FloatTensor([reward]),current_state])
            transition_buffer.push([previous_state, torch.FloatTensor([action_ind]), torch.FloatTensor([reward]), current_state])

            #Updating model
            update_model(model,transition_buffer,optimizer)

            if done is True:
                print("Episode",episode,", steps = ", i)
                break