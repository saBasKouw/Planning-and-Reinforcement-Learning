
from Environment import Environment
from Robot import Robot
import copy

import math
import numpy as np
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import backend as K

import matplotlib.pyplot as plt

STARTING_POS = [0, 3]

LEFT = 'a'
RIGHT = 'd'
UP = 'w'
DOWN = 's'
DIM = 4
ROBOT_SYM = 8

batch_size = 64
done = False


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=4000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.argmax(self.model.predict(next_state)[0]))
            target_future = self.model.predict(state)
            target_future[0][action] = target

            self.model.fit(state, target_future, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)


def init_world():
    return Robot(STARTING_POS[0], STARTING_POS[1]), Environment()


def print_map(env, robot):
    map = copy.deepcopy(env.map)
    map[robot.y][robot.x] = ROBOT_SYM
    [print(line) for line in map]


def move_rob(action, rob):
    if action == 0:
        rob.direction = "left"
        rob.move(-1, 0)
    elif action == 1:
        rob.direction = "right"
        rob.move(+1, 0)
    elif action == 2:
        rob.direction = "up"
        rob.move(0, -1)
    elif action == 3:
        rob.direction = "down"
        rob.move(0, +1)


def slip(rob, env):
    if rob.direction == "left":
        for i in range(rob.x):
            rob.move(-1, 0)
            if env.is_on_crack(rob):
                break
    elif rob.direction == "right":
        for i in range(DIM - rob.x):
            rob.move(+1, 0)
            if env.is_on_crack(rob):
                break
    elif rob.direction == "up":
        for i in range(rob.y):
            rob.move(0, -1)
            if env.is_on_crack(rob):
                break
    elif rob.direction == "down":
        for i in range(DIM - rob.y):
            rob.move(0, +1)
            if env.is_on_crack(rob):
                break


def should_slip(rob, env):
    slip_chance = [0.05, 0.95]
    result = np.random.choice(2, 1, p=slip_chance)
    if result == 0:
        slip(rob, env)


def check_already_been(rob, track):
    for coor in track:
        if coor[0] == rob.x and coor[1] == rob.y:
            return True
    return False

rob, env = init_world()

deaths = 0
reached_goal = 0

n_episodes = 1000
state_size = 4
action_size = 4
scores = []

agent = DQNAgent(state_size, action_size)




for episode in range(n_episodes):
    done = False
    while True:

        state = env.surroundings(rob)
        state = np.reshape(state, [1, state_size])

        action = agent.act(state)
        old_track = rob.track

        move_rob(action, rob)

        if env.is_on_ice(rob):
            should_slip(rob, env)

        next_state = env.surroundings(rob)
        next_state = np.reshape(next_state, [1, state_size])

        distance = math.hypot(3-rob.x, 0-rob.y)

        if env.is_on_crack(rob):
            deaths += 1
            reward = -10
            done = True

        elif env.is_on_goal(rob):
            reached_goal += 1
            reward = 100
            done = True
        elif env.is_on_ship(rob):
            env.map[rob.y][rob.x] = 4
            reward = 20
        elif check_already_been(rob, old_track):
            reward = -20
        else:
            reward = 0

        agent.remember(state, action, reward, next_state, done)

        if done:
            rob, env = init_world()
            break

    #validation below
    if episode % 50 == 0:
        deaths = 0
        reached_goal = 0
        out_time = 0
        for j in range(100):
            done = False
            for i in range(20):
                state = env.surroundings(rob)
                state = np.reshape(state, [1, state_size])

                action = agent.act(state)
                move_rob(action, rob)

                if env.is_on_crack(rob):
                    deaths += 1
                    done = True
                elif env.is_on_goal(rob):
                    reached_goal += 1
                    done = True
                if done:
                    rob, env = init_world()
                    track = rob.track
                    break
                if i == 19:
                    out_time += 1
        print(track)
        print(
            "episode: {}/{}, score: {}, e: {:.6}".format(episode, n_episodes, reached_goal/100, agent.epsilon))
    scores.append(reached_goal / 100)
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)



plt.plot(scores)
plt.show()



