import time
import numpy as np
import jax
import jax.numpy as jnp

from cube_model_naive import Cube
from gather_data import gen_training_set, batch_generator
from cnn import cnn_init as init_fun
from cnn import cnn_apply as apply_fun

from jax.experimental import optimizers
opt_init, opt_update, get_params = optimizers.adam(step_size=0.0001)


def loss(params, batch, apply_fun):
    X, (y_vals, y_acts), w = batch
    vhead_out, phead_out = apply_fun(params, X)

    num_train = X.shape[0]

    mse_loss = np.sum(((vhead_out - y_vals) ** 2).squeeze() * w) / np.sum(w)
    cross_entropy_loss = - np.sum(phead_out[np.arange(num_train), y_acts.squeeze()] * w) / np.sum(w)
    return mse_loss + cross_entropy_loss


@jax.jit
def update(i, opt_state, batch):
    params = get_params(opt_state)
    L, grads = jax.value_and_grad(loss)(params, batch, apply_fun)
    return L, opt_update(i, grads, opt_state)



def train(input_shape, batch_size, num_epochs, seed=None):
    """
    """
    # Initialize pseudo-random number generator.
    rng = jax.random.PRNGKey(seed)

    # Initialize optimizer state.
    rng, init_rng = jax.random.split(rng)
    _, params = init_fun(init_rng, input_shape)
    opt_state = opt_init(params)


    l = 100         # number of episodes
    k_max = 10      # sequence length
    loss_history = []

    # Begin training.
    for epoch in range(num_epochs):
        tic = time.time()

        # Generate data from self-play and build a batch generator.
        rng, data_rng, batch_rng = jax.random.split(rng, 3)
        data = gen_training_set(data_rng, l, k_max, params, apply_fun)
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


if __name__ == "__main__":
    input_shape = (3, 18, 6)
    batch_size = 16
    num_epochs = 3
    seed = 42

    params, loss_history = train(input_shape, batch_size, num_epochs, seed)

#