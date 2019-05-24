class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.track = []
        self.direction = None
        self.reward = 0

    def move(self, x, y):
        self.track.append([self.x, self.y])
        if not (self.x + x == -1 or self.x + x == 4 or
                self.y + y == -1 or self.y + y == 4):
            self.x += x
            self.y += y


