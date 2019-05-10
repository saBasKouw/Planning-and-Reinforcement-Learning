class Environment:
    def __init__(self):
        self.map = [[0, 0, 0, 2],
                    [0, 1, 0, 1],
                    [0, 0, 3, 1],
                    [0, 1, 1, 1]]

    def what_tile(self, robot):
        tile = self.map[robot.y][robot.x]
        if tile == 0:
            return "ice"
        elif tile == 1:
            return "crack"
        elif tile == 2:
            return "goal"
        elif tile == 3:
            return "ship"

    def surroundings(self, robot):
        x, y = robot.x, robot.y
        surroundings = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        result = []
        for coor in surroundings:
            x, y = coor[0], coor[1]
            if x < 0 or x > 3 or y < 0 or y > 3:
                result.append(-1)
            else:
                result.append(self.map[y][x])
        return result


    def is_on_ice(self, rob):
        tile = self.what_tile(rob)
        return tile == "ice"

    def is_on_ship(self, rob):
        tile = self.what_tile(rob)
        return tile == "ship"

    def is_on_goal(self, rob):
        tile = self.what_tile(rob)
        return tile == "goal"

    def is_on_crack(self, rob):
        tile = self.what_tile(rob)
        return tile == "crack"








