
from Assignment1.Environment import Environment
from Assignment1.Robot import Robot
import copy

import numpy as np
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt

STARTING_POS = [0, 3]

LEFT = 'a'
RIGHT = 'd'
UP = 'w'
DOWN = 's'
DIM = 4
ROBOT_SYM = 8

batch_size = 128
done = False


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.99995
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

    def remember(self, state, action, reward, next_state, fin_reward, done):
        self.memory.append((state, action, reward, next_state, fin_reward, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, fin_reward, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * fin_reward)
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



cum_rewards = 0

for episode in range(n_episodes):
    done = False
    reward = 0
    states = []
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

        # distance = math.hypot(3-rob.x, 0-rob.y)

        if env.is_on_crack(rob):
            deaths += 1
            reward -= 10
            done = True

        elif env.is_on_goal(rob):
            reached_goal += 1
            reward += 100
            done = True
        elif env.is_on_ship(rob):
            reward += 20

        else:
            reward += 0

        states.append([state, action, reward, next_state, done])
        # agent.remember(state, action, reward, next_state, done)

        if done:
            cum_rewards += reward
            scores.append(cum_rewards)

            for state in states:
                fin_reward = reward
                agent.remember(state[0], state[1], state[2], state[3], fin_reward, state[4])
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

                act_values = agent.model.predict(state)
                action = np.argmax(act_values[0])
                # action = agent.act(state)
                move_rob(action, rob)

                if env.is_on_crack(rob):
                    deaths += 1
                    done = True
                elif env.is_on_goal(rob):
                    reached_goal += 1
                    done = True
                if done or i == 19:
                    track = rob.track
                    rob, env = init_world()
                    break
        print(track)
        print(
            "episode: {}/{}, score: {}, e: {:.6}".format(episode, n_episodes, reached_goal/100, agent.epsilon))
    scores.append(reached_goal / 100)
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)


print(sum(scores)/len(scores))
plt.plot(scores)
plt.show()



