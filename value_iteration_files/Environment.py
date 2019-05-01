class Environment:
    def __init__(self):
        self.map = [[0, 0, 0, 2],
                    [0, 1, 0, 1],
                    [0, 0, 0, 1],
                    [0, 1, 1, 1]]



    def index_of_state(self,x,y):
        count = 0
        for i in range(y):
            count +=1
        for b in range(x):
            count += 1
        return count

    def what_tile(self, coor):
        tile = self.map[coor[1]][coor[0]]
        if tile == 0:
            return "ice"
        elif tile == 1:
            return "crack"
        elif tile == 2:
            return "goal"
        elif tile == 3:
            return "ship"


    def surroundings_of(self, state):
        count = -1
        for y in range(4):
            for x in range(4):
                count +=1
                if count == state:
                    return [(x-1, y, "left"),
                            (x+1, y, "right"),
                            (x, y-1, "up"),
                            (x, y+1, "down")]

    def get_tile(self, state):
        count = -1
        for y in range(4):
            for x in range(4):
                count +=1
                if count == state:
                    return x, y




    def is_on_ice(self, rob):
        tile = self.what_tile((rob.x, rob.y))
        return tile == "ice"

    def is_on_ship(self, rob):
        tile = self.what_tile((rob.x, rob.y))
        return tile == "ship"

    def is_on_goal(self, rob):
        tile = self.what_tile((rob.x, rob.y))
        return tile == "goal"

    def is_on_crack(self, rob):
        tile = self.what_tile((rob.x, rob.y))
        return tile == "crack"








