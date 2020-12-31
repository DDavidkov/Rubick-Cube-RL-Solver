import time

from cubelet_cube.rubick_env import RubickEnv
from common.move import RubickMove

import jax
from jax.experimental.stax import Dense, Relu, LogSoftmax, serial, Flatten
from jax import random
from jax.experimental import optimizers

from jax import grad, value_and_grad
from jax.tree_util import tree_flatten
import jax.numpy as jnp

import numpy as np

seed = 0
opt_init, opt_update, get_params = optimizers.adam(step_size=0.001)


def fcnn(out_dim=(1, 12)):
    shared_init, shared_apply = serial(Flatten,
                                       Dense(4096),
                                       Relu,
                                       Dense(2048),
                                       Relu,
                                       Dense(512),
                                       Relu)

    v_head_init, v_head_apply = Dense(out_dim[0])
    p_head_init, p_head_apply = serial(Dense(out_dim[1]), LogSoftmax)

    def init_func(rng, input_shape):
        rng, v_head_rng, p_head_rng = jax.random.split(rng, 3)

        shared_out_shape, shared_params = shared_init(rng, input_shape)
        v_head_out_shape, v_head_params = v_head_init(v_head_rng, shared_out_shape)
        p_head_out_shape, p_head_params = p_head_init(p_head_rng, shared_out_shape)

        out_shape = (v_head_out_shape, p_head_out_shape)
        params = [shared_params, v_head_params, p_head_params]

        return out_shape, params

    def apply_func(params, inputs):
        shared_params, v_head_params, p_head_params = params

        shared_out = shared_apply(shared_params, inputs)
        v_head_out = v_head_apply(v_head_params, shared_out)
        p_head_out = p_head_apply(p_head_params, shared_out)

        return v_head_out, p_head_out

    return init_func, apply_func

init, apply = fcnn()

def l2_regularizer(params, alpha):
    return alpha*jnp.sum(jnp.array([jnp.sum(jnp.abs(w)**2) for w in tree_flatten(params)[0] ]))

def loss(params, batch, apply_func):
    X, (y_vals, y_acts), w = batch
    vhead_out, phead_out = apply_func(params, X)

    num_train = X.shape[0]

    mse_loss = jnp.sum(((vhead_out - y_vals) ** 2).squeeze() * w) / jnp.sum(w)
    cross_entropy_loss = - jnp.sum(phead_out[jnp.arange(num_train), y_acts.squeeze()] * w) / jnp.sum(w)
    return mse_loss + cross_entropy_loss + l2_regularizer(params, 0.001)

@jax.jit
def update(i, opt_state, batch):
    params = get_params(opt_state)
    L, grads = value_and_grad(loss)(params, batch, apply)
    return L, opt_update(i, grads, opt_state)

def gen_training_set(rng, episode_count, actions_count, params, apply_fun):
    """
    @params episode_count (int): Number of episodes.
    @params actions_count (int): Maximum number of shuffle steps in each episode.
    """
    states, trg_vals, trg_acts, w = [], [], [], []

    # Create a model environment.
    cube = RubickEnv()

    for i in range(episode_count):
        rng, k_rng, act_rng = jax.random.split(rng, num=3)

        # Generate a random sequence of states starting from the solved state.
        actions = jax.random.randint(act_rng, shape=(actions_count,), minval=0, maxval=12)
        cube.reset()

        episode_history = np.array([cube.step(RubickMove(act), True) for act in actions]).T
        state_history, rewards_history, expand_history = episode_history

        expand_states = np.array([history[0] for history in expand_history])
        expand_rewards = np.array([history[1] for history in expand_history])

        flat_func = lambda arr: arr.reshape(-1, 48)
        flat_expand_history = np.array(list(map(flat_func, expand_states))).reshape(-1, 48)

        # Evaluate children states.
        vals, policies = jax.jit(apply_fun)(params, flat_expand_history)

        # Make targets.
        vals = vals.reshape(actions_count, 12) + expand_rewards
        acts = np.argmax(vals, axis=1)
        vals = np.max(vals, axis=1)

        # Store results.
        states.extend(state_history)
        trg_vals.extend(vals)
        trg_acts.extend(acts)
        w.extend((1 / (d + 1) for d in range(actions_count)))

    # Build the dataset.
    data = {"X": np.stack(states),
            "y": (np.vstack(trg_vals), np.vstack(trg_acts)),
            "w": np.stack(w)}

    return data

def batch_generator(rng, data, batch_size):
    num_train = data["X"].shape[0]
    num_batches = num_train // batch_size + bool(num_train % batch_size)

    perm = jax.random.permutation(rng, num_train)

    for i in range(num_batches):
        idxs = np.array(perm[i * batch_size : (i + 1) * batch_size])
        yield (data["X"][idxs],
               (data["y"][0][idxs], data["y"][1][idxs]),
               data["w"][idxs])

def train(input_shape, batch_size, num_epochs, seed=None):
    """
    """
    # Initialize pseudo-random number generator.
    rng = jax.random.PRNGKey(seed)

    # Initialize optimizer state.
    rng, init_rng = jax.random.split(rng)
    _, params = init(init_rng, input_shape)
    opt_state = opt_init(params)

    l = 100  # number of episodes
    k_max = 10  # sequence length
    loss_history = []

    # Begin training.
    for epoch in range(num_epochs):
        tic = time.time()

        # Generate data from self-play and build a batch generator.
        rng, data_rng, batch_rng = jax.random.split(rng, 3)
        data = gen_training_set(data_rng, l, k_max, params, apply)
        train_batches = batch_generator(batch_rng, data, batch_size)

        mean_loss = 0.0
        for i, batch in enumerate(train_batches):
            loss, opt_state = update(i, opt_state, batch)
            mean_loss = (i * mean_loss + loss) / (i + 1)

        # Book-keeping.
        loss_history.append(mean_loss)

        # Record the time needed for a single epoch.
        toc = time.time()

        # Printout results.
        print("(Epoch ({}, {}) took {:.3f} seconds), mean_loss = {:.3f}".format(
            epoch + 1, num_epochs, (toc - tic), mean_loss))
    params = get_params(opt_state)
    return params, loss_history

MAX_MOVES = 100
CUBES = 300

def test_params(env, params, apply_func):
    level_threshold = CUBES / 3
    successful_solves = 0

    for i in range(CUBES):
        difficulty = int(i // level_threshold + 1)
        env.reset()
        env.scramble(difficulty * 5)

        reward = -1
        moves = 0

        while reward == -1 and moves < MAX_MOVES:
            state = np.array([env.get_nn_state()])
            v_head, p_head = apply_func(params, state)
            action = np.argmax(p_head)
            env_state, reward = env.step(RubickMove(int(action)))
            moves += 1

        if reward == 1:
            successful_solves += 1

    print("Out of {} cubes, the NN was able to solve {} successfully.".format(CUBES, successful_solves))
