import numpy as np
from Assignment1.Environment import Environment
from Assignment1.Robot import Robot
import copy
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

##################RANDOM POLICY AND MOVEMENTS
user_input = input("How many iterations?")
try:
    maxiter = int(user_input)
except ValueError:
    print("That's not a number.")

counter = 0
reward = 0
rob, env = init_world()
deaths = 0
reached_goal = 0
stepcount = 0
totalsteps = 0
while counter < maxiter:
    move_pos = random.choice([LEFT, RIGHT, UP, DOWN])
    move_rob(move_pos, rob)

    if env.is_on_ice(rob):
        should_slip(rob, env)

    if env.is_on_crack(rob):
        deaths += 1
        reward -= 10
        print("DEATH!")
        counter += 1
        totalsteps += stepcount
        stepcount = 1
        rob, env = init_world()
    elif env.is_on_goal(rob):
        reached_goal += 1
        reward += 100
        print("GOAL!")
        # print(rob.track)
        counter += 1
        totalsteps += stepcount
        stepcount = 1
        rob, env = init_world()
    elif env.is_on_ship(rob):
        env.map[rob.y][rob.x] = 4
        reward += 20
        stepcount += 1
    else:
        reward += 0
        stepcount += 1
    if counter < maxiter:
        print("Step ", stepcount)
        print_map(env, rob)

print("The robot has tried the maze ", maxiter, " times using a random policy")
print("And died ", deaths, " times and reached the endgoal ", reached_goal, " times")
print("The average number of steps per iteration of the robot: ", totalsteps / maxiter)
print("The accumulated reward of the robot: ", reward)
