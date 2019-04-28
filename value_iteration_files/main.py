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

##########BELOW IS THE VALUE ITERATION
# def get_actions(env, state):
#     sur = env.surroundings_of(state)
#     value_of_actions = [0, 0, 0, 0]
#
#     if env.what_tile(env.get_tile(state)) == "crack": # if you remove this cracks are good states
#         #  because often the next statesof a crack are better than the next states of ice
#         return value_of_actions
#     for i, next_tile in enumerate(sur):
#         if out_of_bounds(next_tile):
#             value_of_actions[i] = 0
#         else:
#             if env.what_tile(env.get_tile(state)) == "goal": # if you remove this then goal is not a good next state
#                 reward = 100
#             else:
#                 reward = tile_reward(next_tile, env)
#
#             next_state = table[next_tile[1]][next_tile[0]]
#             value_of_actions[i] += 0.95 * (reward + 0.9 * state_values[next_state])
#
#     return value_of_actions
no_probs = {'left':0, 'right':0, 'up':0, 'down':0}
random_probs = {'left':0.25, 'right':0.25, 'up':0.25, 'down':0.25}
current_policy = {0:random_probs.copy(), 1:random_probs.copy(), 2:random_probs.copy(), 3:random_probs.copy(), 4:random_probs.copy(), 5:random_probs.copy(), 6:random_probs.copy(), 7:random_probs.copy(),
                  8:random_probs.copy(), 9:random_probs.copy(), 10:random_probs.copy(), 11:random_probs.copy(), 12:random_probs.copy(), 13:random_probs.copy(), 14:random_probs.copy(), 15:random_probs.copy()}
def policy_evaluation(policy):
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
            break

def policy_iteration():
    changing = False
    for state in range(num_states):
        chosen_a = max(current_policy[state],key=current_policy[state].get)
        sur = env.surroundings_of(state)
        values = calculate_value(state, sur)
        best_a = max(values,key=itemgetter(0))
        if chosen_a != best_a[1]:
            changing = True
            current_policy[state] = no_probs.copy()
            current_policy[state][best_a[1]] = 1
    return changing



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
    reward = tile_reward(env.get_tile(state),env)
    #reward = 0
    value_of_s_primes = [0, 0, 0, 0]
    discount_factor = 0.5

    for i, s_prime in enumerate(arr):
        slip_reward, slip_state = slip_transition(s_prime)

        if env.what_tile(env.get_tile(state)) == "crack" or env.what_tile(env.get_tile(slip_state)) == "crack":
            return calculate_value(12, env.surroundings_of(12))

        if out_of_bounds(s_prime):
            value = 1*(reward + discount_factor * state_values[state]) #Out of bounds direction will have same value for slipping so can be combined into prob 1
        else:
            next_state = table[s_prime[1]][s_prime[0]]
            reward_s_prime = tile_reward(env.get_tile(next_state),env)
            value = 0.95 * (reward_s_prime + discount_factor * state_values[next_state]) + 0.05 *(
                        slip_reward + discount_factor * state_values[slip_state])

        value_of_s_primes[i] = (value,s_prime[2])

    return value_of_s_primes

def policy_iteration_init():
    iteration = 0
    changing = True
    while changing:
        iteration += 1
        policy_evaluation(current_policy)
        changing = policy_iteration()
        print("Policy iteration: ", iteration)
    print("Final policy: ",current_policy)
    print("Optimal path:", get_optimal_path())

def get_optimal_path():
    state = 12
    path = []
    while True:
        sur = env.surroundings_of(state)
        pot_states = []
        for coor in sur:
            if out_of_bounds(coor):
                pot_states.append(0)
            else:
                state = table[coor[1]][coor[0]]
                pot_states.append(state_values[state])
        index = np.argmax(pot_states)
        path_coor = sur[index]
        state = table[path_coor[1]][path_coor[0]]
        path.append(state)
        if state == 3:
            return path

def value_iteration():
    # delta = 0
    # while delta < theta:
    stop_condition = 0.0001
    while True:
        difference = 0
        for state in range(num_states):
            sur = env.surroundings_of(state)
            actions = calculate_value(state, sur)  # gets the values of each action so the value for all 4 next states
            # with exception of state == crack then values are 0 or when state is goal then values are 100
            print(state, actions)
            best_action_value = max([value[0] for value in actions])  # get the value of the best action.
            difference = max(difference, np.abs(best_action_value - state_values[state])) # this delta method did not work for me
            # wonder why
            state_values[state] = best_action_value
        if difference < stop_condition:
            break

    # print(state_values, "\n")

    # #Create the value table
    # value_matrix = list(map(list,table))
    # counter = 0
    # for i in range(4):
    #     for j in range(4):
    #         value_matrix[i][j] = state_values[counter]
    #         counter += 1
    # for i in range(4):
    #     print(value_matrix[i])
    # print(state_values)

    # Check according to the value table what the best path is.


    print("optimal path: ", get_optimal_path())
    print("values", state_values)

value_iteration()