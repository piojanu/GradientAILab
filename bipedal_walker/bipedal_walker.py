import gym
from itertools import count
import torch, torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import random
import math


NUM_EPISODES = 2000
SCREEN_WIDTH = 400
SCREEN_LENGTH =600
WINDOW_MAX_Y = 300
WINDOW_MIN_Y = 200
BUFFER_SIZE = 65536
GAMMA = 0.999
START_EXPLORE_RATIO = 0.7
END_EXPLORE_RATIO = 0.05
NUM_FEATURES = 24
FALL_TIME = 30

# Check if GPU is available
use_gpu = torch.cuda.is_available()

if use_gpu:
    print("Using GPU")
    LongTensor = torch.cuda.LongTensor
    FloatTensor = torch.cuda.FloatTensor

else:
    print("Using CPU")
    LongTensor = torch.LongTensor
    FloatTensor = torch.FloatTensor


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.hidden1 = nn.Linear(NUM_FEATURES, 400)
        self.hidden2 = nn.Linear(400, 300)
        self.output = nn.Linear(300, 16)

        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 'n' is number of inputs to each neuron
                n = len(m.weight.data[1])
                # "Xavier" initialization
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.zero_()

        self.optimizer = optim.RMSprop(self.parameters(), lr=0.0001)

    def forward(self,x):
        x = F.tanh(self.hidden1(x))
        x = F.tanh(self.hidden2(x))
        return self.output(x)

    def update(self,transition_buffer):
        # Compute a mask of non-final states and concatenate the batch elements
        transition_batch = transition_buffer.get_batch()

        #If buffer_size < batch_size
        if transition_batch is None:
            return

        states, actions, rewards, next_states = transition_batch
        non_final_mask = [i for i, state in enumerate(next_states) if state is not None]
        non_final_mask = LongTensor(non_final_mask)
        # We don't want to backprop through the expected action values and volatile
        # will save us on temporarily changing the model parameters'
        # requires_grad to False!
        non_final_next_states = Variable(torch.cat([s for s in next_states if s is not None]).view(-1, NUM_FEATURES),
                                         volatile=True)

        states = Variable(torch.cat(list(states)).view(-1, NUM_FEATURES))
        actions = Variable(torch.cat(list(actions)).view(-1, 1).type(LongTensor))
        rewards = Variable(torch.cat(list(rewards)))

        # print("States:",states,", actions:",actions,", rewards:",rewards)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        q_values = self.forward(states).gather(1, actions)

        # Compute V(s_{t+1}) for all next states. This is equal to taking max of q_values
        next_state_values = Variable(torch.zeros(32).type(FloatTensor))

        next_state_values[non_final_mask] = self.forward(non_final_next_states).max(1)[0]

        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_state_values.volatile = False

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + rewards

        # Compute Huber loss
        loss = F.smooth_l1_loss(q_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class TransitionBuffer:
    def __init__(self):
        self.buffer = []
        self.batch_size = 32

    def push(self,transition):
        self.buffer.append(transition)
        if len(self.buffer) > BUFFER_SIZE:
            self.buffer.pop(0)

    def get_batch(self):
        if len(self.buffer) >= self.batch_size:
            batch = random.sample(self.buffer, self.batch_size)
            # Transposing list of lists
            return list(zip(*batch))


#Returns the index of the action with max value according to our DQN model
def get_action(model,state, explore_ratio, randomization=True):
    chance = random.random()
    if chance < explore_ratio and randomization:
        return LongTensor([random.randint(0, 15)])
    else:
        q_values = model(state)

        return q_values.max(0)[1].data


def get_action_vec(action_ind):
    #Getting binary vector of action ie. 0 is 0000 and 15 is 1111
    action_vec = np.array([int(bit) for bit in '{0:04b}'.format(action_ind)])

    #Changing value scope from {0,1} to {-1,1}
    return action_vec*2 - 1


def get_decay_ratio():
    return math.pow(END_EXPLORE_RATIO/START_EXPLORE_RATIO, 1.5/NUM_EPISODES)

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2').unwrapped
    env.reset()
    model = DQN()

    if use_gpu:
        model.cuda()

    transition_buffer = TransitionBuffer()

    reward_his = np.zeros(NUM_EPISODES)
    steps_his = np.zeros(NUM_EPISODES)
    distance_his = np.zeros(NUM_EPISODES)
    velocity_his = np.zeros(NUM_EPISODES)

    min_max_states = np.zeros((NUM_FEATURES, 2))

    explore_ratio = START_EXPLORE_RATIO
    explore_decay_ratio = get_decay_ratio()

    for episode in range(NUM_EPISODES):
        env.reset()
        action_vec = env.action_space.sample()


        current_state = FloatTensor(np.zeros(NUM_FEATURES))

        for i in count():
        #for i in range(100):
            #Rendering screen
            env.render(mode='rgb_array')

            #Getting action vector[a0,a1,a2,a3], aN = {-1,1}
            if i < FALL_TIME:
                # Forcing the walker to fall in particular position
                action_ind = 8
                #action_vec = np.array([1, -1, -1, -1])

            else:
                randomization = bool(np.mod(episode, 50))

                # Getting best action index
                action_ind = get_action(model, Variable(current_state, volatile=True), explore_ratio, randomization)

                # Getting action for each joint(4 values {-1,1})
                action_vec = get_action_vec(int(action_ind.cpu().numpy()))

            #Executing step according to our action
            obs, reward, done, info = env.step(action_vec)

            #Saving velocity to calculate distance traveled
            distance_his[episode] += obs[2]

            if done is False:
                next_state = FloatTensor(obs[:NUM_FEATURES])
                reward_his[episode] += reward
            else:
                next_state = None

            #Storing transition into transition buffer
            if i >= FALL_TIME:
                transition_buffer.push([current_state, action_ind, FloatTensor([reward]), next_state])

            #Updating states
            current_state = next_state

            #Updating model

            model.update(transition_buffer)

            if done is True:
                steps_his[episode] = i
                velocity_his[episode] = distance_his[episode]/i
                print("Episode", episode, ", steps = ", i,
                      ", total reward:", reward_his[episode],
                      ", steps_avg:", np.mean(steps_his[:episode+1]),
                      ", reward_avg:", np.mean(reward_his[:episode+1]),
                      ", distance traveled:", distance_his[episode],
                      ", average speed:", velocity_his[episode],
                      ", explore ratio:", explore_ratio)
                break

        if explore_ratio > END_EXPLORE_RATIO:
            explore_ratio = explore_ratio*explore_decay_ratio

    plt.ylabel("distance traveled")
    plt.xlabel("episode id")
    plt.plot(np.arange(0, NUM_EPISODES, 1), distance_his)
    plt.show()

    plt.ylabel("avg velocity")
    plt.xlabel("episode id")
    plt.plot(np.arange(0, NUM_EPISODES, 1), velocity_his)
    plt.show()
