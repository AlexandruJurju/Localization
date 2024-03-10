from enum import Enum


class Motion:
    def __init__(self, x_delta: int, y_delta: int):
        self.x_delta = x_delta
        self.y_delta = y_delta


class Movement(Enum):
    STAY = Motion(0, 0)
    RIGHT = Motion(0, 1)
    LEFT = Motion(0, -1)
    DOWN = Motion(1, 0)
    UP = Motion(-1, 0)
