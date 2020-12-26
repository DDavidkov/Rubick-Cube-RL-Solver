from enum import Enum
import numpy as np

map_func = lambda x: [1 if x == i else 0 for i in np.arange(6)]

triple_cubelet_dict = {
    0: [0, 3, 4],
    1: [1, 0, 4],
    2: [2, 1, 4],
    3: [3, 2, 4],
    4: [3, 0, 5],
    5: [0, 1, 5],
    6: [1, 2, 5],
    7: [2, 3, 5]
}

class TC(Enum):
    YGR = 0
    BYR = 1
    WBR = 2
    GWR = 3
    GYO = 4
    YBO = 5
    BWO = 6
    WGO = 7

    def __call__(self):
        return list(map(map_func, triple_cubelet_dict[self.value]))

double_cubelte_dict = {
    0: [4, 0],
    1: [4, 1],
    2: [4, 2],
    3: [4, 3],
    4: [3, 0],
    5: [0, 1],
    6: [1, 2],
    7: [2, 3],
    8: [0, 5],
    9: [1, 5],
    10: [2, 5],
    11: [3, 5]
}

class DC(Enum):
    RY = 0
    RB = 1
    RW = 2
    RG = 3
    GY = 4
    YB = 5
    BW = 6
    WG = 7
    YO = 8
    BO = 9
    WO = 10
    GO = 11

    def __call__(self):
        return list(map(map_func, double_cubelte_dict[self.value]))
