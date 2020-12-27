import numpy as np
import copy
import random
from collections import namedtuple



class Cube:
    """
    A concrete class representation of a Rubik's Cube.
    The cube is represented as a flattened 2D matrix.

          5 5 5                 # 1 is orange
          5 5 5                 # 2 is green
          5 5 5                 # 3 is red
    1 1 1 2 2 2 3 3 3 4 4 4     # 4 is white
    1 1 1 2 2 2 3 3 3 4 4 4     # 5 is blue
    1 1 1 2 2 2 3 3 3 4 4 4     # 6 is yellow
          6 6 6            
          6 6 6            
          6 6 6            

    """

    # 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6    # 4 is white
    # 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6    # 5 is blue
    # 1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6    # 6 is yellow

    #--------  nested named tuples --------#
    Side = namedtuple("Side", ["LEFT", "FRONT", "RIGHT", "BACK", "UP", "DOWN"])(
                                0, 1, 2, 3, 4, 5)
    Direction = namedtuple("Direction", ["Anti_Clockwise", "Clockwise"])(0, 1)
    Color = namedtuple("Color", ["ORANGE", "GREEN", "RED", "BLUE", "WHITE", "YELLOW"])(
                                0, 1, 2, 3, 4, 5)

    #---------- cube initializer ----------#
    def __init__(self):
        """ Initialize a cube object. """
        self._action_space = [(_s, _dir) for _s in self.Side for _dir in self.Direction]

        self._terminal_state = [np.reshape([_col] * 9, (3,3)) for _col in self.Color]
        # self._action_space = [(_s, _dir) for _s in self.Side.get_sides() for _dir in self.Direction.get_directions()]

        # self._terminal_state = [np.reshape([_col] * 9, (3,3)) for _col in self.Color.get_colors()]

        self._state = None
        self.reset()

    #----------- public methods -----------#
    def step(self, action):
        """ Make a single step taking the specified action.
        @param action (int): An integer value in the range [0, 12).
        """
        if action not in self._action_space:
            raise Exception("Unknown action %s", action)

        prev_state = copy.deepcopy(self._state)

        # Extract the side to be rotated and the direction of rotation.
        side, direction = action

        # Rotate the side in the given direction.
        self._state[side] = np.rot90(self._state[side], k=(-1) ** direction)

        # Rotate the adjecent rows.
        if side == self.Side.LEFT:
            if direction == self.Direction.Anti_Clockwise:
                self._state[self.Side.UP].T[0] = prev_state[self.Side.FRONT].T[0]
                self._state[self.Side.FRONT].T[0] = prev_state[self.Side.DOWN].T[0]
                self._state[self.Side.DOWN].T[0] = np.flip(prev_state[self.Side.BACK].T[2])
                self._state[self.Side.BACK].T[2] = np.flip(prev_state[self.Side.UP].T[0])
            else:
                self._state[self.Side.UP].T[0] = np.flip(prev_state[self.Side.BACK].T[2])
                self._state[self.Side.FRONT].T[0] = prev_state[self.Side.UP].T[0]
                self._state[self.Side.DOWN].T[0] = prev_state[self.Side.FRONT].T[0]
                self._state[self.Side.BACK].T[2] = np.flip(prev_state[self.Side.DOWN].T[0])

        if side == self.Side.FRONT:
            if direction == self.Direction.Anti_Clockwise:
                self._state[self.Side.LEFT].T[2] = np.flip(prev_state[self.Side.UP][2])
                self._state[self.Side.DOWN][0] = prev_state[self.Side.LEFT].T[2]
                self._state[self.Side.RIGHT].T[0] = np.flip(prev_state[self.Side.DOWN][0])
                self._state[self.Side.UP][2] = prev_state[self.Side.RIGHT].T[0]
            else:
                self._state[self.Side.LEFT].T[2] = prev_state[self.Side.DOWN][0]
                self._state[self.Side.DOWN][0] = np.flip(prev_state[self.Side.RIGHT].T[0])
                self._state[self.Side.RIGHT].T[0] = prev_state[self.Side.UP][2]
                self._state[self.Side.UP][2] = np.flip(prev_state[self.Side.LEFT].T[2])

        if side == self.Side.RIGHT:
            if direction == self.Direction.Anti_Clockwise:
                self._state[self.Side.FRONT].T[2] = prev_state[self.Side.UP].T[2]
                self._state[self.Side.DOWN].T[2] = prev_state[self.Side.FRONT].T[2]
                self._state[self.Side.BACK].T[0] = np.flip(prev_state[self.Side.DOWN].T[2])
                self._state[self.Side.UP].T[2] = np.flip(prev_state[self.Side.BACK].T[0])
            else:
                self._state[self.Side.FRONT].T[2] = prev_state[self.Side.DOWN].T[2]
                self._state[self.Side.DOWN].T[2] = np.flip(prev_state[self.Side.BACK].T[0])
                self._state[self.Side.BACK].T[0] = np.flip(prev_state[self.Side.UP].T[2])
                self._state[self.Side.UP].T[2] = prev_state[self.Side.FRONT].T[2]

        if side == self.Side.BACK:
            if direction == self.Direction.Anti_Clockwise:
                self._state[self.Side.RIGHT].T[2] = prev_state[self.Side.UP][0]
                self._state[self.Side.DOWN][2] = np.flip(prev_state[self.Side.RIGHT].T[2])
                self._state[self.Side.LEFT].T[0] = prev_state[self.Side.DOWN][2]
                self._state[self.Side.UP][0] = np.flip(prev_state[self.Side.LEFT].T[0])
            else:
                self._state[self.Side.RIGHT].T[2] = np.flip(prev_state[self.Side.DOWN][2])
                self._state[self.Side.DOWN][2] = prev_state[self.Side.LEFT].T[0]
                self._state[self.Side.LEFT].T[0] = np.flip(prev_state[self.Side.UP][0])
                self._state[self.Side.UP][0] = prev_state[self.Side.RIGHT].T[2]

        if side == self.Side.UP:
            if direction == self.Direction.Anti_Clockwise:
                self._state[self.Side.FRONT][0] = prev_state[self.Side.LEFT][0]
                self._state[self.Side.RIGHT][0] = prev_state[self.Side.FRONT][0]
                self._state[self.Side.BACK][0] = prev_state[self.Side.RIGHT][0]
                self._state[self.Side.LEFT][0] = prev_state[self.Side.BACK][0]
            else:
                self._state[self.Side.FRONT][0] = prev_state[self.Side.RIGHT][0]
                self._state[self.Side.RIGHT][0] = prev_state[self.Side.BACK][0]
                self._state[self.Side.BACK][0] = prev_state[self.Side.LEFT][0]
                self._state[self.Side.LEFT][0] = prev_state[self.Side.FRONT][0]

        if side == self.Side.DOWN:
            if direction == self.Direction.Anti_Clockwise:
                self._state[self.Side.FRONT][2] = prev_state[self.Side.RIGHT][2]
                self._state[self.Side.RIGHT][2] = prev_state[self.Side.BACK][2]
                self._state[self.Side.BACK][2] = prev_state[self.Side.LEFT][2]
                self._state[self.Side.LEFT][2] = prev_state[self.Side.FRONT][2]
            else:
                self._state[self.Side.FRONT][2] = prev_state[self.Side.LEFT][2]
                self._state[self.Side.RIGHT][2] = prev_state[self.Side.FRONT][2]
                self._state[self.Side.BACK][2] = prev_state[self.Side.RIGHT][2]
                self._state[self.Side.LEFT][2] = prev_state[self.Side.BACK][2]

        # Check if this is the final state.
        done = self.is_solved()

        if done:
            reward = 1
        else:
            reward = -1

        return copy.deepcopy(self._state), reward, done

    def random_state(self):
        """ Set the current state of the cube to a random state. """
        self.reset()
        for i in range(100):
            action = random.choice(self._action_space)
            self.step(action)

    def reset(self):
        """ Set the current state of the cube to the terminal state. """
        self._state = copy.deepcopy(self._terminal_state)

    def is_valid(self):
        """ Return True if the current state is a valid state for the cube. """
        return True

    def is_solved(self):
        """ Return True if the current state is the terminal state for the cube. """
        return (np.array(self._state) == np.array(self._terminal_state)).all()


    # def matrix_shape(self):
    #     result = np.zeros((9, 12), dtype=int)

    #     for i in range(4):
    #         result[3:6].T[3*i : 3*(i+1)] = self._state[i].T
    #     result[0:3].T[3:6] = self._state[4].T
    #     result[6:9].T[3:6] = self._state[5].T
    #     return result

    # def plot(self):
    #     print(self.matrix_shape)

    def next_state(self, act):
        current_state = copy.deepcopy(self._state)
        next_state, _, _ = self.step(act)
        self._state = current_state
        return next_state
    
    def state(self):
        return _one_hot(np.hstack(self._state).reshape(1, 3, 18, 1), 6)


def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[: np.newaxis] == np.arange(k), dtype)



def plot_state(state):
    result = np.zeros((9, 12), dtype=int)

    for i in range(4):
        result[3:6].T[3*i : 3*(i+1)] = state[i].T
    result[0:3].T[3:6] = state[4].T
    result[6:9].T[3:6] = state[5].T
    return result


def expand_state(state):
    cube = Cube()

    children = []
    rewards = []
    for act in cube._action_space:
        cube._state = copy.deepcopy(state)
        child, reward, _ = cube.step(act)
        children.append(child)
        rewards.append(reward)
    
    return children, rewards

def vectorize_state(state):
    return _one_hot(np.hstack(state).reshape(1, 3, 18, 1), 6)



#