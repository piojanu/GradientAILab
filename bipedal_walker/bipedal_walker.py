import gym
from itertools import count
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt

NUM_EPISODES = 50
SCREEN_WIDTH = 400
SCREEN_LENGTH =600

WINDOW_MAX_Y = 300
WINDOW_MIN_Y = 200

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(448, 4)

    def forward(self,x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

def get_screen():
    screen = env.render(mode='rgb_array')[WINDOW_MIN_Y:WINDOW_MAX_Y]
    screen = screen.transpose(2,0,1) #Change to num_channels, width, height for pytorch
    print(screen.shape)
    #plt.imshow(screen)
    #plt.show()
    return screen

def get_action():
    return env.action_space.sample()

def update_model():
    pass

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2').unwrapped
    env.reset()
    for episode in range(NUM_EPISODES):
        previous_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - previous_screen
        env.reset()
        for i in count():
            current_screen = get_screen()
            action = get_action()

            obs,reward,done,info = env.step(action)

            if done is True:
                break