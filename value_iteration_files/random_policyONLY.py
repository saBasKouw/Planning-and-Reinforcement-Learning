import numpy as np
from Environment import Environment
from Robot import Robot
import copy
from operator import itemgetter
import random

STARTING_POS = [0, 3]

LEFT = 'a'
RIGHT = 'd'
UP = 'w'
DOWN = 's'
DIM = 4
ROBOT_SYM = 8

def init_world():
    return Robot(STARTING_POS[0], STARTING_POS[1]), Environment()


def print_map(env, robot):
    map = copy.deepcopy(env.map)
    map[robot.y][robot.x] = ROBOT_SYM
    [print(line) for line in map]


def move_rob(move, rob):
    if move == LEFT:
        rob.direction = "left"
        rob.move(-1, 0)
    elif move == RIGHT:
        rob.direction = "right"
        rob.move(+1, 0)
    elif move == UP:
        rob.direction = "up"
        rob.move(0, -1)
    elif move == DOWN:
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


rob, env = init_world()

deaths = 0
reached_goal = 0

num_states = 16
state_values = [0,0,0,0,
                0,0,0,0,
                0,0,0,0,
                0,0,0,0]

CURRENT_STATE = 0
theta = 80
table = [[0, 1, 2, 3],
         [4, 5, 6, 7],
         [8, 9, 10, 11],
         [12, 13, 14, 15]]


def out_of_bounds(tile):
    return tile[0] == -1 or tile[0] == 4 or  tile[1] == -1 or tile[1] == 4


def tile_reward(tile, env):
    type_tile = env.what_tile(tile)
    if type_tile == "crack":
        return -10
    elif type_tile == "goal":
        return 100
    elif type_tile == "ship":
        return 20
    else:

        return 0

no_probs = {'left':0, 'right':0, 'up':0, 'down':0}
random_probs = {'left':0.25, 'right':0.25, 'up':0.25, 'down':0.25}
current_policy = {0:random_probs.copy(), 1:random_probs.copy(), 2:random_probs.copy(), 3:random_probs.copy(), 4:random_probs.copy(), 5:random_probs.copy(), 6:random_probs.copy(), 7:random_probs.copy(),
                  8:random_probs.copy(), 9:random_probs.copy(), 10:random_probs.copy(), 11:random_probs.copy(), 12:random_probs.copy(), 13:random_probs.copy(), 14:random_probs.copy(), 15:random_probs.copy()}
def policy_evaluation():
    stop_condition = 0.0001
    iteration = 0
    while True:
        difference = 0
        iteration += 1
        print("Policy evaluation iteration:", iteration)
        for state in range(num_states):
            sur = env.surroundings_of(state)
            values = calculate_value(state,sur)
            summed = sum([value[0]*current_policy[state][value[1]] for value in values])
            difference = max(difference, np.abs(summed - state_values[state]))
            state_values[state] = summed
        if difference < stop_condition:
            print("policy evaluation random policy ", state_values)
            break


def slip_transition(s_prime):
    direction = s_prime[2]
    rob.direction = direction
    rob.x = s_prime[0]
    rob.y = s_prime[1]
    slip(rob, env)
    slip_state = env.index_of_state(rob.x, rob.y)
    slip_reward = tile_reward(env.get_tile(slip_state), env)
    return slip_reward, slip_state

def calculate_value(state, arr):
    reward = tile_reward(env.get_tile(state), env)
    # reward = 0
    value_of_s_primes = [0, 0, 0, 0]
    discount_factor = 0.9
    prob_no_slip = 0.95
    for i, action in enumerate(arr):
        slip_reward, slip_state = slip_transition(action)

        if env.what_tile(env.get_tile(state)) == "crack" or env.what_tile(env.get_tile(state)) == "goal":
            # return calculate_value(12, env.surroundings_of(12))
            value = 0
        elif out_of_bounds(action):
            value = 1 * (0 + discount_factor * state_values[
                state])  # Out of bounds direction will have same value for slipping so can be combined into prob 1
        else:
            next_state = table[action[1]][action[0]]
            reward_s_prime = tile_reward(env.get_tile(next_state), env)
            value = prob_no_slip * (reward_s_prime + discount_factor * state_values[next_state]) + (
                        1 - prob_no_slip) * (
                            slip_reward + discount_factor * state_values[slip_state])

        value_of_s_primes[i] = (value, action[2])

    return value_of_s_primes

#run the evaluation
policy_evaluation()

