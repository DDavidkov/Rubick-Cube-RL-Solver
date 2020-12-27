import random
import itertools
from copy import deepcopy


import numpy as np
import jax
import jax.numpy as jnp


from cube_model_naive import Cube, plot_state, expand_state, vectorize_state


def generate_state_sequence(rng, k, pretty_print=False):
    """
    Generate a random sequence of states starting from the solved state.

    @param k (int): Length of generated sequence.
    @returns states (List[state]): A list of states containing the entire sequence.
    # @returns states (Array): Numpy array of shape (k, H, W, C) containing the
    #                             entire sequence of states.
    """
    cube = Cube()
    actions = np.array(jax.random.randint(rng, shape=(k,), minval=0, maxval=12))
    # states = np.zeros((k, 3, 18, 6))
    states = []

    for i, act in enumerate(actions):
        dir, side = divmod(act, 6)
        state, _, _ = cube.step((side, dir))
        # states[i] = cube.state().copy()
        states.append(state)

        if pretty_print:
            print("Taking action: {} {}\n".format(cube.Side._fields[side],
                                                  cube.Direction._fields[dir]))
            print(plot_state(cube._state))
            print("\n========== #### ==========\n")

    return states


def expand_state_sequence(states):
    """
    Given a sequence of states, for every state visit its one-step-away neighbours.

    @param states (List[state]): A list of states containing the entire sequence.
    # @param states (Array): Numpy array of shape (k, H, W, C) containing the
    #                             entire sequence of states.
    @returns expanded_states (List[state]):
    @returns expanded_rewards (List[int]):
    """
    expanded = [expand_state(s) for s in states]    # -> (states, rewards)

    expanded_states = [s for s,r in expanded]
    expanded_rewards = [r for s,r in expanded]


    expanded_states = list(itertools.chain.from_iterable((expanded_states)))
    expanded_rewards = list(itertools.chain.from_iterable((expanded_rewards)))

    return expanded_states, expanded_rewards


def eval_states(expanded_states, params, apply_fun):
    """
    Run the input states through the model to collect prediction results.

    @param states (List[state]): A list of states.
    # @param states (Array): A numpy array of shape (num_states, H, W, C).
    @param params (List[Array]): Model parameters.
    @param apply_fun (func): Apply function predictiing results.
    @returns preds ():
    """
    exp_st = np.zeros((len(expanded_states), 3, 18, 6))

    for i, state in enumerate(expanded_states):
        exp_st[i] = vectorize_state(state)

    preds = apply_fun(params, exp_st)
    return preds


def make_targets(preds, rewards):
    """
    Given model predictions and action rewards, set value and policy
    targets using Q-learning approach.
    ``` V_target(s_t) = max_a (R(s_t, a) + V_target(s_t+1)) ```
    ``` p_target(s_t) = argmax_a (R(s_t, a) + V_target(s_t+1)) ```

    @params preds ():
    @params rewards (List[int]): A list of rewards.
    @returns vals ():
    @returns acts ():
    """
    vals, policies = preds
    num_train, num_actions = policies.shape
    seq_len = num_train // num_actions
    vals = vals.reshape(seq_len, num_actions)
    rewards = np.array(rewards).reshape(seq_len, num_actions)
    vals += rewards

    acts = np.argmax(vals, axis=1)
    vals = np.max(vals, axis=1)

    return vals, acts


def gen_training_set(rng, l, k_max, params, apply_fun):
    """
    @params l (int): Number of episodes.
    @params k_max (int): Maximum number of shuffle steps in each episode.
    """
    X, values, actions, weights = [], [], [], []

    for i in range(l):
        # rng, k_rng = jax.random.split(rng)
        # k = jax.random.randint(rng, shape=(1,), minval=0, maxval=k_max)
        k = k_max
        states = generate_state_sequence(rng, k)
        exp_states, exp_rewards = expand_state_sequence(states)
        preds = eval_states(exp_states, params, apply_fun)
        vals, acts = make_targets(preds, exp_rewards)

        X.extend(vectorize_state(s) for s in states)
        values.extend(vals)
        actions.extend(acts)
        weights.extend((1 / (d+1) for d in range(k)))

    data = {"X": np.vstack(X),
            "y": (np.vstack(values), np.vstack(actions)),
            "w": np.array(weights)}
    return data


def batch_generator(rng, data, batch_size):
    num_train = data["X"].shape[0]
    num_batches = num_train // batch_size + bool(num_train % batch_size)

    perm = jax.random.permutation(rng, num_train)

    for i in range(num_batches):
        idxs = perm[i * batch_size : (i + 1) * batch_size]
        idxs = np.array(idxs)   ######
        yield (data["X"][idxs],
               (data["y"][0][idxs], data["y"][1][idxs]),
               data["w"][idxs])


if __name__ == "__main__":
    seed = 0
    np.random.seed(seed)
    rng = jax.random.PRNGKey(0)

    # out_dim = (1, 12)
    # cnn_init, cnn_apply = conv_net(out_dim)
    # out_shape, params = cnn_init(rng, (3, 18, 6))
    
    # rng, data_rng, batch_rng = jax.random.split(rng, 3)
    # data = gen_training_set(data_rng, 2, 3, params, cnn_apply)
    # train_batches = batch_generator(batch_rng, data, 4)

    # batch = next(train_batches)
    # print(batch[0].shape)
    # print(batch[1])

    # batch = next(train_batches)
    # print(batch[0].shape)
    # print(batch[1])