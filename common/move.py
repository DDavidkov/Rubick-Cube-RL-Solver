from enum import Enum

class RubickMove(Enum):
    U = 0
    U_PRIM = 1
    D = 2
    D_PRIM = 3
    F = 4
    F_PRIM = 5
    B = 6
    B_PRIM = 7
    R = 8
    R_PRIM = 9
    L = 10
    L_PRIM = 11
    SCRAMBLE = 12
    RESET = 13
    UNDO = 14
    REDO = 15
