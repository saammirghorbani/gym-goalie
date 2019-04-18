
import argparse
import sys

import gym_goalie
import gym
from gym import wrappers, logger
import keras
import random
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Training Parameters
n_episodes=1000
n_win_ticks=195
max_env_steps=40
gamma=1.0 # Discount factor. Consideration of future rewards - same policies (the ones that balance the pole) are optimal
          # at every step.
epsilon=1.0 # Exploration. Agent chooses action that it believes has the best long-term effect with probability 1- epsilon,
            # and it chooses an action uniformly at random, otherwise.
epsilon_min=0.01 # we want to put a little randomness in decision eventually
epsilon_decay=0.995 #we want to decay epsilon as agent stops taking actions randomly
alpha=0.01 # the learning rate, determines to what extent the newly acquired information will override the old information.
alpha_decay=0.01 # we want to decay alpha so that agent will not change the actions/learning everytime even after achieveing enough learning
batch_size=64
monitor=False
quiet=False

env = gym.make('gym_goalie:Goalie-v0')

# Environment Parameters
memory = deque(maxlen=100000)
#env._max_episode_steps = 50



# Model Definition
model = Sequential()
model.add(Dense(256, input_dim=25, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(27, activation='relu'))
model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))


def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


def choose_action(state, epsilon):
    return action_sample() if (np.random.random() <= epsilon) else number_to_action(np.argmax(
        model.predict(state)))


def get_epsilon(t):
    return max(epsilon_min, min(epsilon, 1.0 - math.log10((t + 1) * epsilon_decay)))


def preprocess_state(state):
    return np.reshape(state['observation'], [1, 25])


def action_to_number(action): 
    number = 0
    number += action[0]+1
    number += (action[1]+1)*3
    number += (action[2]+1)*9
    return number


def number_to_action(number):
    action = [0, 0, 0, 0]
    action[2] = math.trunc(number / 9) - 1
    number = number % 9 - 1
    action[1] = math.trunc(number / 3) - 1
    number = number % 3
    action[0] = number - 1
    return action


def action_sample():
    return [np.random.randint(-1, 2), np.random.randint(-1, 2), np.random.randint(-1, 2), 0]


def replay(batch_size, epsilon):
    x_batch, y_batch = [], []
    minibatch = random.sample(
        memory, min(len(memory), batch_size))
    for state, action, reward, next_state, done in minibatch:

        action = action_to_number(action)
        y_target = model.predict(state)

        for i in range(len(y_target)):
            y_target[i] = -y_target[i]

        y_target[0][action] = reward if done else reward + gamma * np.max(model.predict(next_state)[0])

        x_batch.append(state[0])
        y_batch.append(y_target[0])

    model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


def run():
    scores = deque(maxlen=100)

    for e in range(n_episodes):
        state = preprocess_state(env.reset())
        done = False
        i = 0
        while not done:
            action = choose_action(state, get_epsilon(e))
            next_state, reward, done, _ = env.step(action)
            # if e % 20 == 0:
            env.render()
            next_state = preprocess_state(next_state)
            remember(state, action, reward, next_state, done)
            state = next_state
            i += 1

        scores.append(i)
        mean_score = np.mean(scores)
        if mean_score >= n_win_ticks and e >= 100:
            if not quiet: print('Ran {} episodes. Solved after {} trials'.format(e, e - 100))
            return e - 100
        if e % 100 == 0 and not quiet:
            print('[Episode {}] - Mean survival time over last 100 episodes was {} ticks.'.format(e, mean_score))

        replay(batch_size, get_epsilon(epsilon))

    if not quiet: print('Did not solve after {} episodes'.format(e))
    return e


if __name__ == '__main__':
    run()

# class RandomAgent(object):
#     """The world's simplest agent!"""
#     def __init__(self, action_space):
#         self.action_space = action_space
#
#     def act(self, observation, reward, done):
#         # has the form [-1:1, -1:1, -1:1, -1:1]
#         return self.action_space.sample()
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description=None)
#     parser.add_argument('env_id', nargs='?', default='gym_goalie:Goalie-v0', help='Select the environment to run')
#     args = parser.parse_args()
#
#     # You can set the level to logger.DEBUG or logger.WARN if you
#     # want to change the amount of output.
#     logger.set_level(logger.INFO)
#
#
#     env = gym.make('gym_goalie:Goalie-v0')
#
#     # You provide the directory to write to (can be an existing
#     # directory, including one with existing data -- all monitor files
#     # will be namespaced). You can also dump to a tempdir if you'd
#     # like: tempfile.mkdtemp().
#     outdir = '/tmp/random-agent-results'
#     env = wrappers.Monitor(env, directory=outdir, force=True)
#     env.seed(0)
#     agent = RandomAgent(env.action_space)
#
#     episode_count = 100
#     reward = 0
#     done = False
#
#     env.env._max_episode_steps = 40  # not sure what this means? Doesn't seem represent nbr of steps in simulation
#
#     for i in range(episode_count):
#         ob = env.reset()
#         while True:
#             action = agent.act(ob, reward, done)
#             ob, reward, done, _ = env.step(action)
#             env.render()
#             if done:
#                 break
#             # Note there's no env.render() here. But the environment still can open window and
#             # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
#             # Video is not recorded every episode, see capped_cubic_video_schedule for details.
#
#     # Close the env and write monitor result info to disk
#     env.close()