import numpy as np

from common.move import RubickMove

from cubelet_cube.cubelets import TC, DC

class RubickEnv:
    def __init__(self, seed=0):
        self.set_seed(seed)

        self.solved_state = {
            "triples": np.array([
                [TC.YGR(), TC.BYR(), TC.WBR(), TC.GWR()],
                [TC.GYO(), TC.YBO(), TC.BWO(), TC.WGO()]
            ]),
            "doubles": np.array([
                [DC.RY(), DC.RB(), DC.RW(), DC.RG()],
                [DC.GY(), DC.YB(), DC.BW(), DC.WG()],
                [DC.YO(), DC.BO(), DC.WO(), DC.GO()]
            ])
        }

        self.reset()

        self.actions = {
            RubickMove.U: lambda state: self.make_u(state, 1),
            RubickMove.U_PRIM: lambda state: self.make_u(state, -1),
            RubickMove.D: lambda state: self.make_d(state, 1),
            RubickMove.D_PRIM: lambda state: self.make_d(state, -1),
            RubickMove.F: lambda state: self.make_f(state, 1),
            RubickMove.F_PRIM: lambda state: self.make_f(state, -1),
            RubickMove.B: lambda state: self.make_b(state, 1),
            RubickMove.B_PRIM: lambda state: self.make_b(state, -1),
            RubickMove.L: lambda state: self.make_l(state, 1),
            RubickMove.L_PRIM: lambda state: self.make_l(state, -1),
            RubickMove.R: lambda state: self.make_r(state, 1),
            RubickMove.R_PRIM: lambda state: self.make_r(state, -1),
            RubickMove.SCRAMBLE: lambda: self.scramble(100),
            RubickMove.RESET: lambda: self.reset(),
            RubickMove.UNDO: lambda: self.undo(),
            RubickMove.REDO: lambda: self.redo()
        }

    def __repr__(self):
        pretty_state = self.get_pretty_state()
        return pretty_state.__repr__()

    def get_pretty_state(self):
        triples = self.state["triples"].reshape(-1, 3, 6)
        doubles = self.state["doubles"].reshape(-1, 2, 6)

        return np.array([
            [
                [triples[0][0], doubles[0][1], triples[1][1]],
                [doubles[4][1], [1, 0, 0, 0, 0, 0], doubles[5][0]],
                [triples[4][1], doubles[8][0], triples[5][0]]
            ],
            [
                [triples[1][0], doubles[1][1], triples[2][1]],
                [doubles[5][1], [0, 1, 0, 0, 0, 0], doubles[6][0]],
                [triples[5][1], doubles[9][0], triples[6][0]]
            ],
            [
                [triples[2][0], doubles[2][1], triples[3][1]],
                [doubles[6][1], [0, 0, 1, 0, 0, 0], doubles[7][0]],
                [triples[6][1], doubles[10][0], triples[7][0]]
            ],
            [
                [triples[3][0], doubles[3][1], triples[0][1]],
                [doubles[7][1], [0, 0, 0, 1, 0, 0], doubles[4][0]],
                [triples[7][1], doubles[11][0], triples[4][0]]
            ],
            [
                [triples[3][2], doubles[2][0], triples[2][2]],
                [doubles[3][0], [0, 0, 0, 0, 1, 0], doubles[1][0]],
                [triples[0][2], doubles[0][0], triples[1][2]]
            ],
            [
                [triples[4][2], doubles[8][1], triples[5][2]],
                [doubles[11][1], [0, 0, 0, 0, 0, 1], doubles[9][1]],
                [triples[7][2], doubles[10][1], triples[6][2]]
            ]
        ])

    def get_nn_state(self):
        return RubickEnv.transform_state(self.state)

    def scramble(self, number_of_moves=15):
        self.undo_stack.clear()
        self.redo_stack.clear()

        for _ in range(number_of_moves):
            move = RubickMove(np.random.randint(0, 11))
            self.actions[move](self.state)

    def step(self, action: RubickMove, expand=False):
        args = [self.state] if action.value < 12 else []
        self.actions[action](*args)

        if action.value < 12:
            done = RubickEnv.is_solved(self.state, self.solved_state)
            reward = 1 if done else -1

            self.undo_stack.append(RubickEnv.get_opposite_action(action))
            self.redo_stack.clear()

            if expand:
                return self.get_nn_state(), reward, self.expand()
            else:
                return self.get_nn_state(), reward
        else:
            return self.state, None

    def set_seed(self, seed=0):
        self.seed = seed
        np.random.seed(seed)

    def reset(self):
        self.state = {
            "triples": self.solved_state["triples"].copy(),
            "doubles": self.solved_state["doubles"].copy()
        }
        self.undo_stack = []
        self.redo_stack = []

    def can_undo(self):
        return len(self.undo_stack) > 0

    def can_redo(self):
        return len(self.redo_stack) > 0

    def undo(self):
        if self.can_undo():
            action = self.undo_stack.pop()
            self.actions[action](self.state)
            self.redo_stack.append(RubickEnv.get_opposite_action(action))

    def redo(self):
        if self.can_redo():
            action = self.redo_stack.pop()
            self.actions[action](self.state)
            self.undo_stack.append(RubickEnv.get_opposite_action(action))

    def expand(self):
        expanded_states = []
        rewards = []
        for i in range(12):
            state_copy = {
                "triples": self.state["triples"].copy(),
                "doubles": self.state["doubles"].copy()
            }
            self.actions[RubickMove(i)](state_copy)
            reward = 1 if RubickEnv.is_solved(state_copy, self.solved_state) else -1
            rewards.append(reward)
            expanded_states.append(RubickEnv.transform_state(state_copy))

        return np.array(expanded_states), np.array(rewards)

    @staticmethod
    def make_u(state, roll_amount):
        state["triples"][0] = np.roll(state["triples"][0], roll_amount, axis=0)
        state["doubles"][0] = np.roll(state["doubles"][0], roll_amount, axis=0)

    @staticmethod
    def make_d(state, roll_amount):
        state["triples"][1] = np.roll(state["triples"][1], roll_amount, axis=0)
        state["doubles"][2] = np.roll(state["doubles"][2], roll_amount, axis=0)

    @staticmethod
    def make_f(state, roll_amount):
        triples_to_rotate = [(0, 0), (0, 1), (1, 1), (1, 0)]
        RubickEnv.rotate_triple(triples_to_rotate, roll_amount, state["triples"])

        doubles_to_rotate = [(0, 0), (1, 1), (2, 0), (1, 0)]
        RubickEnv.rotate_double(doubles_to_rotate, roll_amount, state["doubles"])

    @staticmethod
    def make_b(state, roll_amount):
        triples_to_rotate = [(0, 2), (0, 3), (1, 3), (1, 2)]
        RubickEnv.rotate_triple(triples_to_rotate, roll_amount, state["triples"])

        doubles_to_rotate = [(0, 2), (1, 3), (2, 2), (1, 2)]
        RubickEnv.rotate_double(doubles_to_rotate, roll_amount, state["doubles"])

    @staticmethod
    def make_l(state, roll_amount):
        triples_to_rotate = [(0, 3), (0, 0), (1, 0), (1, 3)]
        RubickEnv.rotate_triple(triples_to_rotate, roll_amount, state["triples"])

        doubles_to_rotate = [(0, 3), (1, 0), (2, 3), (1, 3)]
        RubickEnv.rotate_double(doubles_to_rotate, roll_amount, state["doubles"])

    @staticmethod
    def make_r(state, roll_amount):
        triples_to_rotate = [(0, 1), (0, 2), (1, 2), (1, 1)]
        RubickEnv.rotate_triple(triples_to_rotate, roll_amount, state["triples"])

        doubles_to_rotate = [(0, 1), (1, 2), (2, 1), (1, 1)]
        RubickEnv.rotate_double(doubles_to_rotate, roll_amount, state["doubles"])

    @staticmethod
    def rotate_double(indexes, roll_amount, original):
        doubles_to_rotate = np.array([original[index] for index in indexes])

        rotated = np.roll(doubles_to_rotate, roll_amount, axis=0)

        for i, index in enumerate(indexes):
            original[index] = rotated[i] if (i % 2 == 0) == (roll_amount == 1) else np.roll(rotated[i], 1, axis=0)

    @staticmethod
    def rotate_triple(indexes, roll_amount, original):
        triples_to_rotate = np.array([original[index] for index in indexes])

        rotated = np.roll(triples_to_rotate, roll_amount, axis=0)

        for i, index in enumerate(indexes):
            direction = -1 if (i % 2 == 0) == (roll_amount == 1) else 1
            original[index] = np.roll(rotated[i], roll_amount * direction, axis=0)

    @staticmethod
    def get_opposite_action(action: RubickMove):
        if action.value > 11:
            return None
        else:
            return RubickMove(action.value + 1) if action.value % 2 == 0 else RubickMove(action.value - 1)

    @staticmethod
    def transform_state(state):
        triples = state["triples"].reshape(-1, 6)
        doubles = state["doubles"].reshape(-1, 6)

        return np.concatenate((triples, doubles))

    @staticmethod
    def is_solved(state, solved_state):
        return ((state["triples"] == solved_state["triples"]).all() and
                (state["doubles"] == solved_state["doubles"]).all())
