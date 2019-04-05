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

while True:
    move = input("Use WASD keys to move: ")

    move_rob(move, rob)

    if env.is_on_ice(rob):
        should_slip(rob, env)

    if env.is_on_crack(rob):
        deaths += 1
        rob.reward -= 10
        rob, env = init_world()
    elif env.is_on_goal(rob):
        reached_goal += 1
        rob.reward += 100
        print(rob.track)
        rob, env = init_world()
    elif env.is_on_ship(rob):
        env.map[rob.y][rob.x] = 4
        rob.reward += 20
    else:
        rob.reward += 1

    print_map(env, rob)
    print(rob.reward)






