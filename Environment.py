import Actions

class Environment:
    def __init__(self):
        # 0 ice
        # 1 is crack
        # 2 is goal
        # 3 is start
        # 4 is ship
        self.world = [[0, 0, 0, 2],
                      [0, 1, 0, 1],
                      [0, 0, 4, 1],
                      [3, 1, 1, 1]]

        self.actions = Actions()

    def which_tile(self, coordinate):
        return self.world[coordinate[1]][coordinate[0]]


    def move(self, move, coordinate):
        if move == "left":
            return "left", [coordinate[0]-1, coordinate[1]]
        elif move == "right":
            return [coordinate[0]+1, coordinate[1]]
        elif move == "up":
            return [coordinate[0], coordinate[1]-1]
        elif move == "down":
            return [coordinate[0], coordinate[1]+1]





