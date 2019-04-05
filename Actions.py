import numpy as np
# # 0 is normal ground
#         # 1 is ice
#         # 2 is crack
#         # 3 is goal
#         # 4 is start
#         # 5 is ship
#         self.world = [[0, 0, 0, 3],
#                       [0, 2, 0, 2],
#                       [0, 0, 5, 2],
#                       [4, 2, 2, 2]]
starting_point = [4,0]

def on_ice(position):
    slip = [0.05,0.95]
    result = np.random.choice(2,1,p=slip)
    if result == 1:
        return position
    if result == 0:
        newposition = world[0][0]

def on_crack(position):
    new_position = starting_point
    return new_position

def out_grid(position):
    new_position = old_position
    return new_position

def on_ship(position):
    return position

on_ice()