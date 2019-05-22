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
action_dict = {0: "left", 1: "right", 2: "up", 3: "down"}

memory = []

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
        return True
    return False

def slip_transition(s_prime):
    direction = s_prime[2]
    rob.direction = direction
    rob.x = s_prime[0]
    rob.y = s_prime[1]
    slip(rob, env)
    slip_state = env.index_of_state(rob.x, rob.y)
    slip_reward = tile_reward(env.get_tile(slip_state), env)
    return slip_reward, slip_state


def init_world():
    return Robot(STARTING_POS[0], STARTING_POS[1]), Environment()


rob, env = init_world()

num_states = 16
num_actions = 4
qtable = np.zeros((num_states, num_actions))

alpha = 0.1
gamma = 0.8
epsilon = 0.1


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

def replay(batch_size=10):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, new_state in minibatch:
        max = np.max(qtable[new_state])
        value = qtable[state, action]
        newval = value + alpha * (reward + (gamma * max) - value)
        qtable[state][action] = newval


def run_episodes(softmax_enabled=False, experience_replay=False):
    state = 12
    ship_taken = False
    for i in range(10000):
        while True:
            if not softmax_enabled:
                if random.uniform(0, 1) < epsilon:
                    # Check the action space
                    action = np.random.choice(4)
                else:
                    action = np.argmax(qtable[state])
            else:
                action = softmax(state)[0]
            rob.direction = action_dict[action]
            if not should_slip(rob,env):
                move_rob(action, rob)
            new_state = env.index_of_state(rob.x, rob.y)
            #print(new_state)
            reward = tile_reward((rob.x, rob.y), env, ship_taken)

            if experience_replay:
                memory.append([state, action, reward, new_state])
            else:
                max = np.max(qtable[new_state])
                value = qtable[state, action]
                newval = value + alpha * (reward + (gamma * max) - value)
                qtable[state][action] = newval

            state = new_state
            print(state)
            if state == 3 or env.what_tile((rob.x,rob.y)) == "crack":
                if experience_replay:
                    replay()

                # print(np.argmax(qtable[10]))
                state = 12
                rob.x = STARTING_POS[0]
                rob.y = STARTING_POS[1]
                rob.direction = ""
                break


def run_episodesSARSA():
    state = 12
    ship_taken = False
    for i in range(10000):
        if random.uniform(0, 1) < epsilon:
            # Check the action space
            action = np.random.choice(4)
        else:
            action = np.argmax(qtable[state])
        while True:
            move_rob(action, rob)
            new_state = env.index_of_state(rob.x, rob.y)
            reward = tile_reward((rob.x, rob.y), env, ship_taken)
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
            print(state)
            if state == 3 or env.what_tile((rob.x,rob.y)) == "crack":
                state = 12
                rob.x = STARTING_POS[0]
                rob.y = STARTING_POS[1]
                rob.direction = ""
                break

def softmax(state):
    probs = np.zeros(4)
    temp = 0.5
    for i in range(4):
        probs[i] = np.exp((qtable[state][i]/temp))
    probs = probs/sum(probs)
    return np.random.choice(4,1,p=probs)

def get_path():
    state = 12
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
run_episodes(softmax_enabled=False, experience_replay=True)
get_path()