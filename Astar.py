"""Implementation of A* heuristic search algorithm"""
import numpy as np
import heapq
import jax
from collections import deque
from functools import partial
from time import time

from src.cube_model_naive import Cube, expand_state, plot_state
from src.cnn import cnn_init, cnn_apply


def astar(state, params, approximator):
    # (value, steps, stateID, actions)
    v, _ = approximator(params, np.array([state]))
    uid = 0
    node_states = {uid: state}
    node_values = {uid: float(v)}
    H = [(-float(v), 0, uid, tuple())]
    heapq.heapify(H)

    _cube = Cube()
    i = 0
    while i < 1000:
        _, steps, current_id, actions = heapq.heappop(H)
        current = node_states[current_id]
        children, rewards = expand_state(current)
        children_values, _ = approximator(params, children)
        for cval, cstate, cact in zip(children_values, children, range(12)):
            uid += 1
            node_states[uid] = cstate
            node_values[uid] = (-float(cval) + steps + 1)
            c_actions = actions + (cact,)
            if _cube.is_terminal(cstate):
                return c_actions
            heapq.heappush(H, (node_values[uid], steps + 1, uid, c_actions))
        i += 1
    return ()

np.random.seed(24)
p = jax.numpy.load('params_cnn.npy', allow_pickle=True)

taken = []
states = []
actions = []
for _ in range(10):
    cube = Cube()
    act = np.random.randint(0, 12, 12)
    for a in act:
        cube.step(a)
    actions.append(act)
    states.append(cube._state)
    taken.append(astar(cube._state, p, cnn_apply))


def plot_path(start_state, actions):
    cube = Cube()
    cube._state = start_state
    for a in actions:
        cube.step(a)
    plot_state(cube._state)


# start = time()
# res = astar(cube._state, p, cnn_apply)
# end = time()
# res, end - start


def shuffled(moves: int):
    cube = Cube()
    actions = np.random.randint(0, 12, moves)
    for a in actions:
        cube.step(a)
    return cube._state