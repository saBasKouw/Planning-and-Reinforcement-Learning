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








