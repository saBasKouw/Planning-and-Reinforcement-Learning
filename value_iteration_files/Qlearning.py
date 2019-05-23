import numpy as np
from value_iteration_files.Environment import Environment
from Robot import Robot
import random
import matplotlib.pyplot as plt
import time as t
from collections import deque
from collections import defaultdict

STARTING_POS = [0, 3]

LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
DIM = 4
ROBOT_SYM = 8
action_dict = {0: "left", 1: "right", 2: "up", 3: "down"}

memory = deque(maxlen=20)
# memory = []

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
    result = np.random.choice(2,1, p=slip_chance)
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
#epsilon = 0.1


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
    if len(memory) > batch_size:
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, new_state in minibatch:
            max = np.max(qtable[new_state])
            value = qtable[state, action]
            newval = value + alpha * (reward + (gamma * max) - value)
            qtable[state][action] = newval

rewards = []
def run_episodes(softmax_enabled=False, experience_replay=False,epsilon=0.1, temperature = 0.5):
    state = 12
    ship_taken = False
    cumulative_reward = 0
    penalty = 0
    for i in range(10000):
        while True:
            if not softmax_enabled:
                if random.uniform(0, 1) < epsilon:
                    # Check the action space
                    action = np.random.choice(4)
                else:
                    action = np.argmax(qtable[state])
            else:
                action = softmax(state,temperature)
            rob.direction = action_dict[action]
            if not should_slip(rob,env):
                move_rob(action, rob)
            new_state = env.index_of_state(rob.x, rob.y)
            #print(new_state)
            reward = tile_reward((rob.x, rob.y), env, ship_taken)
            cumulative_reward += reward
            if reward == -10:
                penalty += 1
            if experience_replay:
                memory.append([state, action, reward, new_state])
            else:
                max = np.max(qtable[new_state])
                value = qtable[state, action]
                newval = value + alpha * (reward + (gamma * max) - value)
                qtable[state][action] = newval

            state = new_state
            if state == 3 or env.what_tile((rob.x,rob.y)) == "crack":
                if experience_replay:
                    replay()
                rewards.append(cumulative_reward)
                # print(np.argmax(qtable[10]))
                state = 12
                rob.x = STARTING_POS[0]
                rob.y = STARTING_POS[1]
                rob.direction = ""
                break

    return cumulative_reward, penalty


def run_episodesSARSA(epsilon=0.1):
    state = 12
    ship_taken = False
    cumulative_reward = 0
    penalty = 0
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
            cumulative_reward += reward


            if experience_replay:
                memory.append([state, action, reward, new_state])

            if reward == -10:
                penalty += 1
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
                rewards.append(cumulative_reward)
                state = 12
                rob.x = STARTING_POS[0]
                rob.y = STARTING_POS[1]
                rob.direction = ""
                break
    return cumulative_reward, penalty

def softmax(state,temp=0.5):
    probs = np.zeros(4)
    for i in range(4):
        probs[i] = np.exp((np.array(qtable[state][i])/temp))
    probs = probs/sum(probs)
    print(probs)
    return np.random.choice(4,p=probs)

def get_path():
    state = 12
    path = []
    while state != 3:
        max = np.argmax(qtable[state])
        move_direction = action_dict[max]
        location = [item for item in env.surroundings_of(state) if item[2]==move_direction][0]
        new_state = env.index_of_state(location[0],location[1])
        path.append(new_state)
        # print("Index: ",state, "Q value: ", qtable[state])
        state = new_state
    return path

lamb = 0.95


def run_episodesET(episodes, epsilon):
    eligibility = defaultdict(lambda: np.zeros(num_actions))
    qtable = defaultdict(lambda: np.zeros(num_actions))
    ship_taken = False
    cumulative_reward = 0
    penalty = 0
    for episode in range(episodes):
        state = 12
        if random.uniform(0, 1) < epsilon:
            action = np.random.choice(4)
        else:
            action = np.argmax(qtable[state])
        while True:
            rob.direction = action_dict[action]
            if not should_slip(rob, env):
                move_rob(action, rob)
            new_state = env.index_of_state(rob.x, rob.y)
            if random.uniform(0, 1) < epsilon:
                 next_action = np.random.choice(4)
            else:
                 next_action = np.argmax(qtable[new_state])
            reward = tile_reward((rob.x, rob.y), env, ship_taken)
            cumulative_reward += reward
            max = np.argmax(qtable[new_state])
            red_val = reward + gamma * qtable[new_state][max]
            td_error = red_val - qtable[state][action]
            # increment the trace to the corresponding state and action with 1
            eligibility[state][action] += 1
            for state in qtable:
                for a in range(len(qtable[state])):

                    # Update Q-table value based on eligibility trace
                    qtable[state][action] += alpha * td_error * eligibility[state][action]

                    # Decay eligibility trace if action is equal to the max (the best action)
                    if next_action is max:
                        eligibility[state][action] = lamb * gamma * eligibility[state][action]
                    # Reset eligibility trace if random action taken, which will differ depending on the chosen epsilon
                    else:
                        eligibility[state][action] = 0
            state = new_state
            action = next_action
            if state == 3 or env.what_tile((rob.x, rob.y)) == "crack":
                rewards.append(cumulative_reward)
                state = 12
                rob.x = STARTING_POS[0]
                rob.y = STARTING_POS[1]
                rob.direction = ""
                break
    state = 12
    path = []
    while state != 3:
        max = np.argmax(qtable[state])
        move_direction = action_dict[max]
        location = [item for item in env.surroundings_of(state) if item[2] == move_direction][0]
        new_state = env.index_of_state(location[0], location[1])
        path.append(new_state)
        print("Index: ", state, "Q value: ", qtable[state])
        state = new_state
    #print(path)

    return cumulative_reward, penalty, path

#run_episodes(softmax_enabled=False)
# run_episodes(softmax_enabled=False, experience_replay=True)
# get_path()


cuml_rewards_per_e = []
cuml_rewards = []
paths = []
penaltys = []
epsilon_tests = [0.1,0.3,0.5]
temp_tests = [0.5,0.8]
for e in epsilon_tests:
    for i in range(1):
        t1 = t.time()
        cuml_reward, penalty = run_episodes(softmax_enabled=False, experience_replay=True,epsilon=e)
        #cuml_reward, penalty = run_episodesSARSA(epsilon=e)
        #cuml_reward, penalty, path = run_episodesET(10000, epsilon=e)
        t2 = t.time()
        print("Time elapsed:",t2-t1)


        cuml_rewards.append(cuml_reward)
        penaltys.append(penalty)
        #cuml_rewards.append(run_episodesSARSA())

        #change this to 'path' if using ET (which is a return value of the run_episodesET method)
        paths.append(get_path())


    # cuml_rewards_per_e.append(sum(cuml_rewards)/len(cuml_rewards))
    # cuml_rewards = []
    plt.plot(rewards,label=e)
    rewards = []
    print(rewards)
plt.legend()
plt.savefig("plots_qlearn_eligibilitytraces2.png")

print(cuml_rewards)
# print(cuml_rewards_per_e)
print(paths)
# print(sum(cuml_rewards)/len(cuml_rewards))
print(sum(penaltys)/len(penaltys))

#run_episodesET(10000, 0.3)