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

def on_ice(position,transform,points):
    slip = [0.05,0.95]
    result = np.random.choice(2,1,p=slip)
    if result == 1:
        return position,points
    if result == 0:
        max_y = 3
        max_x = 3
        if "DOWN":
            position[0] = max_y
        elif "UP":
            position[0] = 0
        elif "RIGHT":
            position[1] = max_x
        elif "LEFT":
            points[1] = 0
        return position,points

def on_crack(position,points):
    new_position = starting_point
    new_points = points-10
    return new_position,new_points

def out_grid(position,points):
    new_position = old_position
    new_points = points
    return new_position,new_points

def on_ship(position,points):
    new_points = points+20
    return position, new_points

def on_goal(position,points):
    new_points = points+100
    return position, new_points
