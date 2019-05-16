import numpy as np
from value_iteration_files.Environment import Environment
from Robot import Robot
import random

STARTING_POS = [0, 3]

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
DIM = 4
ROBOT_SYM = 8

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


def init_world():
    return Robot(STARTING_POS[0], STARTING_POS[1]), Environment()


rob, env = init_world()

num_states = 16
num_actions = 4
qtable = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.8
epsilon = 0.5


def tile_reward(tile, env, ship_taken):
    type_tile = env.what_tile(tile)
    if type_tile == "crack":
        return -10
    elif type_tile == "goal":
        return 100
    elif type_tile == "ship" and not ship_taken:
        return 20
    else:

        return 0


def run_episodes(softmax_enabled):
    state = 12
    ship_taken = False
    for i in range(1000):
        while True:
            if not softmax_enabled:
                if random.uniform(0, 1) > epsilon:
                    # Check the action space
                    action = np.random.choice(4)
                else:
                    action = np.argmax(qtable[state])
            else:
                action = softmax(state)
            move_rob(action, rob)
            new_state = env.index_of_state(rob.x, rob.y)
            reward = tile_reward((rob.x, rob.y), env, ship_taken)
            if new_state == 10:
                ship_taken = True
            max = np.max(qtable[new_state])
            value = qtable[state, action]
            newval = value + alpha * (reward+(gamma*max)-value)
            qtable[state][action] = newval
            state = new_state
            if state == 3 or env.what_tile((rob.x,rob.y)) == "crack":
                # print(np.argmax(qtable[10]))
                state = 12
                rob.x = STARTING_POS[0]
                rob.y = STARTING_POS[1]
                rob.direction = ""
                break


def run_episodesSARSA():
    state = 12
    ship_taken = False
    for i in range(1000):
        if random.uniform(0, 1) > epsilon:
            # Check the action space
            action = np.random.choice(4)
        else:
            action = np.argmax(qtable[state])
        while True:
            move_rob(action, rob)
            new_state = env.index_of_state(rob.x, rob.y)
            reward = tile_reward((rob.x, rob.y), env, ship_taken)
            if new_state == 10:
                ship_taken = True
            if random.uniform(0, 1) < epsilon:
                # Check the action space
                next_action = np.random.choice(4)
            else:
                next_action = np.argmax(qtable[new_state])
            value = qtable[state, action]
            next_value = qtable[new_state,next_action]
            #newval = (1 - alpha) * value + alpha * (reward + gamma * next_action)
            newval = value + alpha * (reward + (gamma * next_value) - value)
            qtable[state][action] = newval
            state = new_state
            action = next_action
            if state == 3 or env.what_tile((rob.x,rob.y)) == "crack":
                state = 12
                rob.x = STARTING_POS[0]
                rob.y = STARTING_POS[1]
                rob.direction = ""
                break

def softmax(state):
    probs = np.zeros(4)
    temp = 1
    for i in range(4):
        probs[i] = np.exp((qtable[state][i]/temp))
    probs = probs/sum(probs)
    return np.random.choice(4,1,p=probs)

def get_path():
    state = 12
    action_dict = {0:"left",1:"right",2:"up",3:"down"}
    path = []
    while state != 3:
        max = np.argmax(qtable[state])
        move_direction = action_dict[max]
        location = [item for item in env.surroundings_of(state) if item[2]==move_direction][0]
        new_state = env.index_of_state(location[0],location[1])
        path.append(new_state)
        print("Index: ",state, "Q value: ", qtable[state])
        state = new_state
    print(path)


#run_episodes(softmax_enabled=False)
run_episodesSARSA()
get_path()