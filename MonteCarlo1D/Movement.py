from enum import Enum


class Motion:
    def __init__(self, x_delta: int):
        self.x_delta = x_delta


class Movement(Enum):
    STAY = Motion(0)
    RIGHT = Motion(1)
    LEFT = Motion(-1)
