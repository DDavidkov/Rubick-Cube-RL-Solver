import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from functools import partial
from math import ceil


# from fcnn import fc_net as model_fn
from cnn import conv_net as model_fn



#-------------------- data generation utilities --------------------#
def expand_states(states, env):
    """ Given an array of states use the model of the environment to
    obtain the descendants of these states and their respective rewards.
    Return the descendants and the rewards.

    @param states (Array[state]): A numpy array of valid states of the environment.
                                  The shape of the array is (N, state.shape),
                                  where N is the number of states.
    @param env (Cube Object): A Cube object representing the environment.
    @returns children (Array[state]): A numpy array giving the children of the input states
                                      The shape of the array is (N * num_acts, state.shape).
    @returns rewards (Array): A numpy array of shape (N, num_acts, 1) containing the
                              respective rewards.
    """
    zipped = [env.expand_state(s) for s in states]
    children, rewards = list(zip(*zipped))
    children = np.vstack(children)
    rewards = np.stack(rewards)
    return children, rewards


def generate_episodes(rng, env, episodes, k):
    """ Generate a random sequence of states starting from the solved state.

    @param env (Cube Object): A Cube object representing the environment.
    @param episodes (int): Number of episodes to be created.
    @param k (int): Length of backward moves.
    @returns states (Array[state]): Sequence of generated states. The shape of the array
                                    is (episodes * k_max, state.shape).
    @returns weights (Array): Array of weights. w[i] corresponds to the weight of states[i].
    @returns children (Array[state]): Sequence of states corresponding to the children of
                                      each of the generated states. The shape of the array
                                      is (episodes * k_max * num_acts, state.shape).
    @returns rewards (Array): Array of rewards. rewards[i] corresponds to the immediate
                              reward on transition to state children[i]
    """
    states, w = [], []

    # Create an environtment.
    cube = env()
    num_actions = cube.num_actions

    # Create `episodes` number of episodes.
    for _ in range(episodes):
        cube.reset()
        rng, sub_rng = jax.random.split(rng)
        actions = jax.random.randint(sub_rng, shape=(k,), minval=0, maxval=num_actions)
        states.extend((cube.step(act)[0] for act in actions))
        w.extend((1 / (d+1) for d in range(k)))

    # Expand each state to obtain children and rewards.
    children, rewards = expand_states(states, env)

    return jnp.array(states), jnp.array(w), jnp.array(children), jnp.array(rewards)


def make_targets(children, rewards, params):
    """ Generate target values.

    @param children (Array[state]): An array giving the children of the input states
                                    The shape of the array is (N * num_acts, state.shape).
    @param rewards (Array): A numpy array of shape (N, num_acts, 1) containing the
                            respective rewards.
    @param params (pytree): Model parameters for the prediction function.
    @returns vals (Array): An array giving the predicted values of each state.
    """
    # Run the states through the network in batches.
    batch_size = 1000
    vals = []
    for i in range(ceil(children.shape[0] / batch_size)):
        v = apply_fun(params, children[i * batch_size : (i + 1) * batch_size])
        vals.append(v)

    # Add rewards to make target values.
    vals = jnp.vstack(vals).reshape(rewards.shape) + rewards
    return jnp.max(vals, axis=1)


def batch_generator(rng, data, batch_size):
    """ Yields random batches of data.

    @param data (Dict):
    @param batch_size (int):
    @yields batch (Dict): Random batch of data of size `batch_size`.
    """
    num_train = data["X"].shape[0]
    while True:
        rng, sub_rng = jax.random.split(rng)
        idxs = jax.random.choice(sub_rng, jnp.arange(num_train), shape=(batch_size,), replace=False)
        yield (data["X"][idxs],
               data["y"][idxs],
               data["w"][idxs])



#------------------------ utility functions ------------------------#
def fib(n, memo={}):
    """ Return the n-th Fibonacci number. """
    if n == 0 or n == 1:
        memo[n] = 1
    elif n not in memo:
        memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]


def reverse_fib(n):
    """ Return the index of the greatest number from the Fibonacci sequence,
    that is smaller than or equal to n. """
    i = 0
    while fib(i+1) <= n:
        i += 1
    return i



#-------------------- optimizer and LR schedule --------------------#
step_size = 1e-3
decay_rate = 0.0
decay_steps = 1
step_fn = optimizers.inverse_time_decay(step_size=step_size,
                                        decay_rate=decay_rate,
                                        decay_steps=decay_steps)
opt_init, opt_update, get_params = optimizers.adam(step_size=step_fn)



#-------------------- params training utilities --------------------#
init_fun, apply_fun = model_fn()


@jax.jit
def l2_regularizer(params, reg=1e-4):
    """ Return the L2 regularization loss. """
    leaves, _ = tree_flatten(params)
    return reg * sum(jnp.vdot(x, x) for x in leaves)


@jax.jit
def loss_fn(params, batch):
    """ Return the total loss computed for a given batch. """
    X, y, w = batch
    vals = apply_fun(params, X)
    mse_loss = jnp.sum(((vals - y) ** 2).squeeze() * w)
    l2_loss = l2_regularizer(params)
    return mse_loss + l2_loss


@jax.jit
def update(i, opt_state, batch):
    """ Perform backpropagation and parameter update. """
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    # clipped_grads = optimizers.clip_grads(grads, clip_norm)   # norm clipping produces very big differences when jit.
    clipped_grads = tree_map(lambda w: jnp.clip(w, -10.0, 10.0), grads)
    return loss, opt_update(i, clipped_grads, opt_state)


def train(rng, env, batch_size=128, num_epochs=5, num_iterations=21,
          num_samples=100, print_every=10, episodes=100, k_min=1, k_max=25,
          verbose=False, params_save_path=None):
    """
    Train the model function by generating simulations of random-play.
    On every epoch generate a new simulation and run multiple iterations.
    On every iteration evaluate the targets using the most recent model parameters
    and run multiple times through the dataset.
    At the end of every epoch check the performance and store the best performing params.
    If the performance drops then decay the step size parameter.

    @param rng (PRNGKey): A pseudo-random number generator.
    @param env (Cube Object): A Cube object representing the environment.
    @param batch_size (int): Size of minibatches used to compute loss and gradient during training.         [optional]
    @param num_epochs (int): The number of epochs to run for during training.                               [optional]
    @param num_iterations (int): The number of iterations through the generated episodes.                   [optional]
    @param num_samples (int): The number of times the dataset is reused.                                    [optional]
    @param print_every (int): An integer. Training progress will be printed every `print_every` iterations. [optional]
    @param episodes (int): Number of episodes to be created.                                                [optional]
    @param k_min (int): Minimum length of sequence of backward moves.                                       [optional]
    @param k_max (int): Maximum length of sequence of backward moves.                                       [optional]
    @param clip_norm (float): A scalar for gradient clipping.                                               [optional]
    @param verbose (bool): If set to false then no output will be printed during training.                  [optional]
    @param params_save_path (str): File path to save the model parameters.                                  [optional]
    @returns params (pytree): The best model parameters obatained during training.                          [optional]
    @returns loss_history (List): A list containing the mean loss computed during each iteration.           [optional]
    """
    loss_history = []

    # Initialize model parameters and optimizer state.
    rng, init_rng = jax.random.split(rng)
    input_shape = env.terminal_state.shape
    params = None
    if params_save_path is None:
        _, params = init_fun(init_rng, input_shape)
    else:
        params = list(jnp.load(params_save_path, allow_pickle=True))

    # Begin training.
    for e in range(num_epochs):
        tic = time.time()
        opt_state = opt_init(params)

        # Generate data from random-play using the environment.
        rng, sub_rng = jax.random.split(rng)
        # k = max(k_min, min(k_max, reverse_fib(e + 1)))  # Slowly increase the length of each episode starting from k_min.
        k = k_max
        states, w, children, rewards = generate_episodes(sub_rng, env, episodes, k)

        # Train the model on the generated data. Periodically recompute the target values.
        epoch_mean_loss = 0.0
        for it in range(num_iterations):
            tic_it = time.time()

            # Make targets for the generated episodes using the most recent params and build a batch generator.
            params = get_params(opt_state)
            tgt_vals = make_targets(children, rewards, params)
            data = {"X" : states, "y" : tgt_vals, "w" : w}
            rng, sub_rng = jax.random.split(rng)
            train_batches = batch_generator(sub_rng, data, batch_size)

            # Run through the dataset and update model params.
            total_loss = 0.0
            for i in range(num_samples):
                batch = next(train_batches)
                loss, opt_state = update(it* num_samples + i, opt_state, batch)
                total_loss += loss

            # Book-keeping.
            iter_mean_loss = total_loss / num_samples
            epoch_mean_loss = (it * epoch_mean_loss + iter_mean_loss) / (it + 1)
            loss_history.append(iter_mean_loss)

            # Printout results.
            toc_it = time.time()
            if it % print_every == 0 and verbose:
                print("\t(Iteration({}/{}) took {:.3f} seconds) iter_mean_loss = {:.3f}".format(
                                                        it + 1, num_iterations, (toc_it-tic_it), iter_mean_loss))
        # Store the model parameters.
        if params_save_path is not None:
            jnp.save(params_save_path, params)

        # Record the time needed for a single epoch.
        toc = time.time()

        # Printout results.
        if verbose:
            print("(Epoch ({}/{}) took {:.3f} seconds), epoch_mean_loss = {:.3f}".format(
                                                        e + 1, num_epochs, (toc-tic), epoch_mean_loss))
    return params, loss_history



if __name__ == "__main__":
    from cube_model_naive import Cube as env

    seed = 0
    rng = jax.random.PRNGKey(seed=seed)

    ### Run training.
    params, loss_history = train(rng, env,
                                 batch_size=128,
                                 num_epochs=1,
                                 num_iterations=1,
                                 num_samples=50,
                                 print_every=1,
                                 episodes=1000,
                                 k_max=15,
                                 verbose=True,
                                 params_save_path=None)

#