import time
import numpy as np
import jax
import jax.numpy as jnp


from cube_model_naive import Cube, plot_state
from gather_data import gen_training_set, batch_generator
from cnn import cnn_init as init_fun
from cnn import cnn_apply as apply_fun


from jax.experimental import optimizers
opt_init, opt_update, get_params = optimizers.adam(step_size=0.001)


def loss(params, batch, apply_fun):
    X, (y_vals, y_acts), w = batch
    vhead_out, phead_out = apply_fun(params, X)
    num_train = X.shape[0]
    mse_loss = np.sum(((vhead_out - y_vals) ** 2).squeeze() * w)# / np.sum(w)
    cross_entropy_loss = - np.sum(phead_out[np.arange(num_train), y_acts.squeeze()] * w)# / np.sum(w)
    return mse_loss + cross_entropy_loss

@jax.jit
def update(i, opt_state, batch, clip_norm):
    params = get_params(opt_state)
    L, grads = jax.value_and_grad(loss)(params, batch, apply_fun)
    grads = optimizers.clip_grads(grads, clip_norm)
    return L, opt_update(i, grads, opt_state)

def train(input_shape, batch_size, num_epochs, num_iterations, params_file, seed=None):
    """
    """
    # Initialize pseudo-random number generator.
    rng = jax.random.PRNGKey(seed)

    # Initialize optimizer state.
    rng, init_rng = jax.random.split(rng)
    _, params = init_fun(init_rng, input_shape)
    opt_state = opt_init(params)

    l = 4000            # number of episodes
    k_max = 25          # sequence length
    clip_norm = 5.0     # gradient clipping
    loss_history = []

    # Begin training.
    for epoch in range(num_epochs):
        tic = time.time()

        # Generate data from self-play.
        rng, data_rng = jax.random.split(rng)
        data = gen_training_set(data_rng, l, k_max, params, apply_fun)

        epoch_mean_loss = 0.0
        for it in range(num_iterations):
            # Build a batch generator.
            rng, batch_rng = jax.random.split(rng)
            train_batches = batch_generator(batch_rng, data, batch_size)
            iter_mean_loss = 0.0
            for i, batch in enumerate(train_batches):
                loss, opt_state = update(i, opt_state, batch, clip_norm)
                iter_mean_loss = (i * iter_mean_loss + loss) / (i + 1)

            if it % 10 == 0:
                print("\t(Iteration({}/{})), iter_mean_loss = {:.3f}".format(
                                                        it + 1, num_iterations, iter_mean_loss))

            epoch_mean_loss = (it * epoch_mean_loss + iter_mean_loss) / (it + 1)

        # Update model params. Save params for later use.
        params = get_params(opt_state)
        jax.numpy.save(params_file, params)

        # Book-keeping.
        loss_history.append(epoch_mean_loss)

        # Record the time needed for a single epoch.
        toc = time.time()

        # Printout results.
        print("(Epoch ({}/{}) took {:.3f} seconds), epoch_mean_loss = {:.3f}".format(
                                                    epoch + 1, num_epochs, (toc - tic), epoch_mean_loss))

    return params, loss_history


if __name__ == "__main__":
    input_shape = (3, 18, 6)
    batch_size = 128
    num_epochs = 100
    num_iterations = 51
    seed = 42

    params, loss_history = train(input_shape, batch_size, num_epochs, num_iterations, seed)

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
