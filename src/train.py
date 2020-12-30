import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten

from cube_model_naive import Cube, plot_state
from gather_data import gen_training_set, batch_generator
from cnn import cnn_init as init_fun
from cnn import cnn_apply as apply_fun


from jax.experimental import optimizers
opt_init, opt_update, get_params = optimizers.adam(step_size=1e-3)


def l2_regularizer(params, reg):
    return reg*jnp.sum(jnp.array([jnp.sum(jnp.abs(w)**2) for w in tree_flatten(params)[0] ]))

def loss(params, reg, batch, apply_fun):
    X, (y_vals, y_acts), w = batch
    vhead_out, phead_out = apply_fun(params, X)
    num_train = X.shape[0]
    mse_loss = np.sum(((vhead_out - y_vals) ** 2).squeeze() * w)# / np.sum(w)
    cross_entropy_loss = - np.sum(phead_out[np.arange(num_train), y_acts.squeeze()] * w)# / np.sum(w)
    l2_loss = l2_regularizer(params, reg)
    return mse_loss + cross_entropy_loss + l2_loss

@jax.jit
def update(i, opt_state, batch, reg, clip_norm):
    params = get_params(opt_state)
    L, grads = jax.value_and_grad(loss)(params, reg, batch, apply_fun)
    grads = optimizers.clip_grads(grads, clip_norm)
    return L, opt_update(i, grads, opt_state)

def train(seed, save_params_file=None, **kwargs):
    """
    """
    input_shape = (3, 18, 6)

    # Unpack keyword arguments.
    batch_size = kwargs.pop("batch_size", 128)
    num_epochs = kwargs.pop("num_epochs", 10)
    num_iterations = kwargs.pop("num_iterations", 10)
    episodes = kwargs.pop("episodes", 10)
    k_max = kwargs.pop("k_max", 10)
    reg = kwargs.pop("reg", 0.0)
    clip_norm = kwargs.pop("clip_norm", np.inf)

    # Initialize pseudo-random number generator.
    rng = jax.random.PRNGKey(seed)

    # Initialize optimizer state.
    rng, init_rng = jax.random.split(rng)
    _, params = init_fun(init_rng, input_shape)
    opt_state = opt_init(params)

    loss_history = []
    data = None

    # Begin training.
    for epoch in range(num_epochs):
        tic = time.time()

        # Generate data from self-play.
        rng, data_rng = jax.random.split(rng)
        data = gen_training_set(data_rng, episodes, k_max, params, apply_fun)

        epoch_mean_loss = 0.0
        for it in range(num_iterations):
            # Build a batch generator.
            rng, batch_rng = jax.random.split(rng)
            train_batches = batch_generator(batch_rng, data, batch_size)
            iter_mean_loss = 0.0
            for i, batch in enumerate(train_batches):
                loss, opt_state = update(i, opt_state, batch, reg, clip_norm)
                iter_mean_loss = (i * iter_mean_loss + loss) / (i + 1)

            if it % 10 == 0:
                print("\t(Iteration({}/{})), iter_mean_loss = {:.3f}".format(
                                                        it + 1, num_iterations, iter_mean_loss))

            epoch_mean_loss = (it * epoch_mean_loss + iter_mean_loss) / (it + 1)

        # Update model params.
        params = get_params(opt_state)

        # Save params for later use.
        if save_params_file is not None:
            jax.numpy.save(save_params_file, params)

        # Book-keeping.
        loss_history.append(epoch_mean_loss)

        # Record the time needed for a single epoch.
        toc = time.time()

        # Printout results.
        print("(Epoch ({}/{}) took {:.3f} seconds), epoch_mean_loss = {:.3f}".format(
                                                    epoch + 1, num_epochs, (toc - tic), epoch_mean_loss))

    return params, loss_history, data


if __name__ == "__main__":
    seed = 42
    save_params_file = None# "params"

    batch_size = 128
    num_epochs = 1
    num_iterations = 101
    episodes = 1
    k_max = 25
    reg = 1e-4
    clip_norm = 5.0

    params, loss_history, data = train(seed, save_params_file,
                                 batch_size=batch_size,
                                 num_epochs=num_epochs,
                                 num_iterations=num_iterations,
                                 episodes=episodes,                 # number of episodes for generating the training set
                                 k_max=k_max,                       # sequence length of each episode
                                 reg=reg,                           # regularization strength
                                 clip_norm=clip_norm)               # max norm of gradients

    # cube = Cube()
    # cube.set_random_state()
    # for _ in range(10):
    #     val, policy = apply_fun(params, np.array([cube._state]))
    #     act = int(np.argmax(policy, axis=1))
    #     cube.step(act)

    #     print("taking action: ", act)
    #     plot_state(cube._state)
    #     print("\n========== #### ===========\n")

#
