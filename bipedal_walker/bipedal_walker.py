import gym
from itertools import count

NUM_EPISODES = 50

def get_screen():
    return env.render(mode='rgb_array')

def get_action():
    return env.action_space.sample()

if __name__ == '__main__':
    env = gym.make('BipedalWalker-v2').unwrapped
    env.reset()
    for episode in range(NUM_EPISODES):
        previous_screen = get_screen()
        current_screen = get_screen()
        env.reset()
        for i in count():
            current_screen = get_screen()
            action = get_action()

            obs,reward,done,info = env.step(action)

            if done is True:
                break