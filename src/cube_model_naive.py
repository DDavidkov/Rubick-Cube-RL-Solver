import numpy as np
from collections import namedtuple


class Cube:
    """
    A concrete class representation of a Rubik's Cube.
    The cube is represented as a flattened 2D matrix.

    1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6     # 1 is orange
    1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6     # 2 is green  
    1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6     # 3 is red
                                            # 4 is white
                                            # 5 is blue
                                            # 6 is yellow
    """
    #--------  nested named tuples --------#
    Side = namedtuple("Side", ["LEFT", "FRONT", "RIGHT", "BACK", "UP", "DOWN"])(
                                0, 1, 2, 3, 4, 5)
    Direction = namedtuple("Direction", ["ANTI_CLOCK", "CLOCK"])(0, 1)
    Color = namedtuple("Color", ["ORANGE", "GREEN", "RED", "BLUE", "WHITE", "YELLOW"])(
                                0, 1, 2, 3, 4, 5)

    #---------- cube initializer ----------#
    def __init__(self):
        """ Initialize a cube object. """
        self._num_actions = len(self.Side) * len(self.Direction)
        self._action_space = np.arange(self._num_actions, dtype=int)
        self._terminal_state = np.hstack([np.full((3,3), _col) for _col in self.Color])
        self._terminal_state = one_hot(self._terminal_state)

        self._state = None
        self.reset()

        self._take_action = {0: self._left_anticlock,
                             1: self._left_clock,
                             2: self._front_anticlock,
                             3: self._front_clock,
                             4: self._right_anticlock,
                             5: self._right_clock,
                             6: self._back_anticlock,
                             7: self._back_clock,
                             8: self._up_anticlock,
                             9: self._up_clock,
                             10: self._down_anticlock,
                             11: self._down_clock}

    #----------- public methods -----------#
    def step(self, act):
        """ Make a single step taking the specified action.

        @param act (int): An integer value in the range [0, 12).
        @returns next_state (state): The next observed state after taking action `act`.
        @returns reward (int): An integer representing the reward after arriving at the next state.
        @returns done (bool): A boolen indicating whether this is a terminal state.
        """
        if act not in self._action_space:
            raise Exception("Unknown action %s", act)

        # Observe the next state after taking action `act`.
        next_state = self._take_action[act]()

        # Change the current state.
        self._state = next_state.copy()

        # Check if this is the final state.
        done = self.is_solved()
        reward = 1 if done else -1

        return next_state, reward, done

    def set_random_state(self):
        """ Set the current state of the cube to a random valid state. """
        self.reset()
        for _ in range(100):
            act = np.random.randint(low=0, high=12)
            self.step(act)

    def reset(self):
        """ Set the current state of the cube to the terminal state. """
        self._state = self._terminal_state.copy()

    def is_valid(self):
        """ Return True if the current state is a valid state for the cube. """
        print("Not implemented!")
        return True

    def is_solved(self):
        """ Return True if the current state is the terminal state for the cube. """
        return np.all(self._state == self._terminal_state)
    
    def is_terminal(self, state):
        """ Return True if the state is the terminal state for the cube. """
        return np.all(state == self._terminal_state)

    #----------- private methods ----------#
    def _left_anticlock(self):
        """ Perform anti-clockwise rotation of the left side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = l, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * u] = self._state[:, 3 * f]
        next_state[:, 3 * f] = self._state[:, 3 * d]
        next_state[:, 3 * d] = self._state[:, 3 * b + 2][::-1]      # flip
        next_state[:, 3 * b + 2] = self._state[:, 3 * u][::-1]      # flip

        return next_state

    def _left_clock(self):
        """ Perform clockwise rotation of the left side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = l, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * f] = self._state[:, 3 * u]
        next_state[:, 3 * d] = self._state[:, 3 * f]
        next_state[:, 3 * u] = self._state[:, 3 * b + 2][::-1]      # flip
        next_state[:, 3 * b + 2] = self._state[:, 3 * d][::-1]      # flip

        return next_state

    def _front_anticlock(self):
        """ Perform anti-clockwise rotation of the front side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = f, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * l + 2] = self._state[-1, 3 * u : 3 * (u + 1)][::-1]   # flip
        next_state[0, 3 * d : 3 * (d + 1)] = self._state[:, 3 * l + 2] 
        next_state[:, 3 * r] = self._state[0, 3 * d : 3 * (d + 1)][::-1]        # flip
        next_state[-1, 3 * u : 3 * (u + 1)] = self._state[:, 3 * r] 

        return next_state

    def _front_clock(self):
        """ Perform clockwise rotation of the front side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = f, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * l + 2] = self._state[0, 3 * d : 3 * (d + 1)] 
        next_state[0, 3 * d : 3 * (d + 1)] = self._state[:, 3 * r][::-1]        # flip
        next_state[:, 3 * r] = self._state[-1, 3 * u : 3 * (u + 1)] 
        next_state[-1, 3 * u : 3 * (u + 1)] = self._state[:, 3 * l + 2][::-1]   # flip

        return next_state

    def _right_anticlock(self):
        """ Perform anti-clockwise rotation of the right side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = r, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * f + 2] = self._state[:, 3 * u + 2] 
        next_state[:, 3 * d + 2] = self._state[:, 3 * f + 2] 
        next_state[:, 3 * b] = self._state[:, 3 * d + 2][::-1]      # flip
        next_state[:, 3 * u + 2] = self._state[:, 3 * b][::-1]      # flip

        return next_state

    def _right_clock(self):
        """ Perform clockwise rotation of the right side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = r, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * f + 2] = self._state[:, 3 * d + 2] 
        next_state[:, 3 * d + 2] = self._state[:, 3 * b][::-1]      # flip
        next_state[:, 3 * b] = self._state[:, 3 * u + 2][::-1]      # flip
        next_state[:, 3 * u + 2] = self._state[:, 3 * f + 2] 

        return next_state

    def _back_anticlock(self):
        """ Perform anti-clockwise rotation of the back side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = b, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * r + 2] = self._state[0, 3 * u : 3 * (u + 1)] 
        next_state[-1, 3 * d : 3 * (d + 1)] = self._state[:, 3 * r + 2] 
        next_state[:, 3 * l] = self._state[-1, 3 * d : 3 * (d + 1)][::-1]       # flip
        next_state[0, 3 * u : 3 * (u + 1)] = self._state[:, 3 * l][::-1]        # flip

        return next_state

    def _back_clock(self):
        """ Perform clockwise rotation of the back side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = b, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows and columns.
        next_state[:, 3 * r + 2] = self._state[-1, 3 * d : 3 * (d + 1)][::-1]   # flip
        next_state[-1, 3 * d : 3 * (d + 1)] = self._state[:, 3 * l] 
        next_state[:, 3 * l] = self._state[0, 3 * u : 3 * (u + 1)][::-1]        # flip
        next_state[0, 3 * u : 3 * (u + 1)] = (self._state[:, 3 * r + 2]) 

        return next_state

    def _up_anticlock(self):
        """ Perform anti-clockwise rotation of the up side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = u, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows.
        next_state[0, 0:12] = np.roll(self._state[0, 0:12], shift=3, axis=0)

        return next_state

    def _up_clock(self):
        """ Perform clockwise rotation of the up side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = u, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows.
        next_state[0, 0:12] = np.roll(self._state[0, 0:12], shift=-3, axis=0)

        return next_state

    def _down_anticlock(self):
        """ Perform anti-clockwise rotation of the down side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = d, a_cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows.
        next_state[-1, 0:12] = np.roll(self._state[-1, 0:12], shift=-3, axis=0)

        return next_state

    def _down_clock(self):
        """ Perform clockwise rotation of the down side.
        Return the resulting state.
        """
        # Unpack cube sides and rotation directions.
        l, f, r, b, u, d = self.Side
        a_cl, cl = self.Direction
        next_state = self._state.copy()

        # Rotate the side in the given direction
        side, dir = d, cl
        next_state[:, 3*side:3*(side+1)] = np.rot90(self._state[:, 3*side:3*(side+1)], k=(-1)**dir)

        # Rotate the adjecent rows.
        next_state[-1, 0:12] = np.roll(self._state[-1, 0:12], shift=3, axis=0)

        return next_state


##############################
def plot_state(state):
    result = np.zeros((9, 12), dtype=int)
    state = np.argmax(state, axis=-1)
    result[3:6, :] = state[:, :12]
    result[0:3, 3:6] = state[:, 12:15]
    result[6:9, 3:6] = state[:, 15:]
    print(result)

def expand_states(states):
    zipped = [expand_state(s) for s in states]
    children, rewards = list(zip(*zipped))
    children = np.stack(children)
    rewards = np.stack(rewards)
    _k, _a, *view = children.shape
    children = children.reshape(_k * _a, *view)
    return children, rewards

def expand_state(state):
    cube = Cube()
    children = []
    rewards = []
    for act in cube._action_space:
        cube._state = state
        child, reward, _ = cube.step(act)
        children.append(child)
        rewards.append(reward)
    return np.stack(children), np.stack(rewards)

def one_hot(state):
    return np.array(np.expand_dims(state, -1) == np.arange(6), dtype=float)

#
