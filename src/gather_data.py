import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from math import ceil


# def expand_state(state, env):
#     """ Given a state use the model of the environment to obtain
#     the descendants of that state and their respective rewards.
#     Return the descendants and the rewards.

#     @param state (Array): A valid state of the environment.
#     @param env (Cube Object): A Cube object representing the environment.
#     @returns children (Array[state]): A numpy array of shape (num_acts, state.shape) giving
#                                       the children of the input state.
#     @returns rewards (Array): A numpy array of shape (num_acts, 1) containing the
#                               respective rewards.
#     """
#     cube = env()
#     children = []
#     rewards = []
#     for act in cube.action_space:
#         cube._state = state
#         child, reward, _ = cube.step(act)
#         children.append(child)
#         rewards.append(reward)
#     return np.stack(children), np.vstack(rewards)   # rewards shape is (num_acts, 1)


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

def generate_episodes(rng, env, episodes, k_max):
    """ Generate a random sequence of states starting from the solved state.

    @param env (Cube Object): A Cube object representing the environment.
    @param episodes (int): Number of episodes to be created.
    @param k_max (int): Maximum length of backward moves.
    @returns states (): Sequence of states.
    """
    states, w = [], []

    # Create an environtment.
    cube = env()
    num_actions = cube.num_actions

    # Create `episodes` number of episodes.
    for _ in range(episodes):
        cube.reset()
        rng, _, act_rng = jax.random.split(rng, num=3)
        # k = int(jax.random.randint(k_rng, shape=(1,), minval=1, maxval=k_max))
        k = k_max
        actions = jax.random.randint(act_rng, shape=(k,), minval=0, maxval=num_actions)
        states.extend((cube.step(act)[0] for act in actions))
        w.extend((1 / (d+1) for d in range(k)))

    # Expand each state to obtain children and rewards.
    children, rewards = expand_states(states, env)

    return jnp.array(states), jnp.array(w), jnp.array(children), jnp.array(rewards)

def make_targets(children, rewards, params, apply_fun):
    """ Generate target values.

    @param children (Array[state]): A numpy array giving the children of the input states
                                    The shape of the array is (N * num_acts, state.shape).
    @param rewards (Array): A numpy array of shape (N, num_acts, 1) containing the
                            respective rewards.
    @param params (pytree): Model parameters for the prediction function.
    @param apply_fun (func): Prediction function.
    @returns vals ():
    @returns acts ():
    """
    # Run the states through the network in batches.
    batch_size = 1000
    vals = []
    for i in range(ceil(children.shape[0] / batch_size)):
        v, _ = apply_fun(params, children[i * batch_size : (i + 1) * batch_size])
        vals.append(v)
    # Add rewards to make target values.
    vals = jnp.vstack(vals).reshape(rewards.shape) + rewards
    return jnp.max(vals, axis=1), jnp.argmax(vals, axis=1)

def batch_generator(rng, data, batch_size):
    """ Yields random batches of data.

    @param data ():
    @param batch_size (int):
    @yields batch ():
    """
    num_train = data["X"].shape[0]
    while True:
        rng, sub_rng = jax.random.split(rng)
        idxs = jax.random.choice(sub_rng, jnp.arange(num_train), shape=(batch_size,), replace=False)
        yield (data["X"][idxs],
               (data["y"][0][idxs], data["y"][1][idxs]),
               data["w"][idxs])



# def gen_training_set(rng, episodes, k_max, params, apply_fun):
#     """
#     @params l (int): Number of episodes.
#     @params k_max (int): Maximum number of shuffle steps in each episode.
#     """
#     states, tgt_vals, tgt_acts, w = [], [], [], []
#     num_actions = 12

#     # Create a model environtment.
#     cube = Cube()

#     jit_apply_fun = jax.jit(apply_fun)

#     # Create `episodes` number of episodes.
#     for _ in range(episodes):
#         rng, _, act_rng = jax.random.split(rng, num=3)
#         # k = int(jax.random.randint(k_rng, shape=(1,), minval=1, maxval=k_max))
#         k = k_max

#         # Generate a random sequence of states starting from the solved state.
#         actions = jax.random.randint(act_rng, shape=(k,), minval=0, maxval=num_actions)
#         cube.reset()
#         state_sequence = [cube.step(act)[0] for act in actions]

#         # Expand each state to obtain children and rewards.
#         children, rewards = expand_states(state_sequence)

#         # Evaluate children states.
#         vals, _ = jit_apply_fun(params, children)

#         # Make targets.
#         vals = vals.reshape(k, num_actions) + rewards
#         acts = np.argmax(vals, axis=1)
#         vals = np.max(vals, axis=1)

#         # Store results.
#         states.extend(state_sequence)
#         tgt_vals.extend(vals)
#         tgt_acts.extend(acts)
#         w.extend((1 / (d+1) for d in range(k)))

#     # Build the dataset.
#     data = {"X": np.stack(states),
#             "y": (np.vstack(tgt_vals), np.vstack(tgt_acts)),
#             "w": np.stack(w)}

#     # Assign value of 0 to all terminal states.
#     # idxs = [cube.is_terminal(s) for s in states]
#     # data["y"][0][idxs] = 0
#     # print(len(states))
#     # tgt_vals = jax.ops.index_update(tgt_vals, jax.ops.index[idxs, :], 0)

#     return data





# def batch_generator(rng, data, batch_size):
#     num_train = data["X"].shape[0]
#     num_batches = num_train // batch_size + bool(num_train % batch_size)
#     perm = jax.random.permutation(rng, num_train)

#     for i in range(num_batches):
#         idxs = perm[i * batch_size : (i + 1) * batch_size]
#         idxs = np.array(idxs)   ######
#         yield (data["X"][idxs],
#                (data["y"][0][idxs], data["y"][1][idxs]),
#                data["w"][idxs])


if __name__=="__main__":
    rng = jax.random.PRNGKey(0)
    from cnn import conv_net
    import time
    from cube_model_naive import Cube
    init_fun, apply_fun = conv_net()
    _, params = init_fun(rng, (3, 18, 6))
    # data = gen_training_set(rng, 2, 4, params, apply_fun)

    tic = time.time()
    states, w, children, rewards = generate_episodes(rng, Cube, 100, 5)
    toc = time.time()
    v, a = make_targets(children, rewards, params, apply_fun)
    tac = time.time()

    print("episode generation takes {:.3f} seconds".format(toc - tic))
    print("making targets takes {:.3f} seconds".format(tac - toc))
    print("values.shape:", v.shape, a.shape)
    print("valus sum:", jnp.sum(v), jnp.sum(a))

#     train_batches = batch_generator(rng, data, 4)
# #