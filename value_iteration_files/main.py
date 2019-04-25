import numpy as np
from Environment import Environment
from Robot import Robot
import copy

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
                print("ok")
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
def get_actions(env, state):
    sur = env.surroundings_of(state)
    value_of_actions = [0, 0, 0, 0]

    if env.what_tile(env.get_tile(state)) == "crack": # if you remove this cracks are good states
        #  because often the next statesof a crack are better than the next states of ice
        return value_of_actions
    for i, next_tile in enumerate(sur):
        if out_of_bounds(next_tile):
            value_of_actions[i] = 0
        else:
            if env.what_tile(env.get_tile(state)) == "goal": # if you remove this then goal is not a good next state
                reward = 100
            else:
                reward = tile_reward(next_tile, env)

            next_state = table[next_tile[1]][next_tile[0]]
            value_of_actions[i] += 0.95 * (reward + 0.9 * state_values[next_state])

    return value_of_actions

# delta = 0
# while delta < theta:
for i in range(3): # value 3 is enough to converge
    # delta = 0
    for state in range(num_states):


        actions = get_actions(env, state) # gets the values of each action so the value for all 4 next states
        # with exception of state == crack then values are 0 or when state is goal then values are 100

        best_action_value = max(actions) # get the value of the best action.

        # delta = max(delta, np.abs(best_action_value - state_values[state])) # this delta method did not work for me
        # wonder why

        state_values[state] = best_action_value



print(state_values, "\n")


# Create the value table
value_table = table
counter = 0
for i in range(4):
    for j in range(4):
        value_table[i][j] = state_values[counter]
        counter += 1
for i in range(4):
    print(value_table[i])


# Check according to the value table what the best path is.
state = 12
path = []
while True:
    sur = env.surroundings_of(state)
    pot_states = []
    for coor in sur:
        if out_of_bounds(coor):
            pot_states.append(0)
        else:
            pot_states.append(value_table[coor[1]][coor[0]])
    index = np.argmax(pot_states)
    path_coor = sur[index]
    state = table[path_coor[1]][path_coor[0]]
    path.append(state)
    if state == 3:
        break

print("optimal path: ", path)