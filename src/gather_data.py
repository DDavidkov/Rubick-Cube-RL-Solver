import jax
import numpy as np


from cube_model_naive import Cube, plot_state, expand_states


def gen_training_set(rng, l, k_max, params, apply_fun):
    """
    @params l (int): Number of episodes.
    @params k_max (int): Maximum number of shuffle steps in each episode.
    """
    states, trg_vals, trg_acts, w = [], [], [], []
    num_actions = 12

    # Create a model environtment.
    cube = Cube()

    # Create `l` number of episodes.
    for i in range(l):
        rng, k_rng, act_rng = jax.random.split(rng, num=3)
        # k = int(jax.random.randint(k_rng, shape=(1,), minval=0, maxval=k_max))
        k = k_max

        # Generate a random sequence of states starting from the solved state.
        actions = jax.random.randint(act_rng, shape=(k,), minval=0, maxval=num_actions)
        cube.reset()
        state_sequence = [cube.step(act)[0] for act in actions]

        # Expand each state to obtain children and rewards.
        children, rewards = expand_states(state_sequence)

        # Evaluate children states.
        vals, policies = jax.jit(apply_fun)(params, children)

        # Make targets.
        vals = vals.reshape(k, num_actions) + rewards
        acts = np.argmax(vals, axis=1)
        vals = np.max(vals, axis=1)

        # Store results.
        states.extend(state_sequence)
        trg_vals.extend(vals)
        trg_acts.extend(acts)
        w.extend((1 / (d+1) for d in range(k)))

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
        idxs = perm[i * batch_size : (i + 1) * batch_size]
        idxs = np.array(idxs)   ######
        yield (data["X"][idxs],
               (data["y"][0][idxs], data["y"][1][idxs]),
               data["w"][idxs])

#