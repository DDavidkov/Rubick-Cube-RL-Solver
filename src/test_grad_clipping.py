import jax
import jax.numpy as jnp
from jax.experimental.stax import serial, Dense, Relu, Flatten
from jax.experimental.optimizers import sgd, adam, clip_grads, l2_norm
from jax.tree_util import tree_map

init_fun, apply_fun = serial(Flatten, Dense(16), Relu, Dense(16),
                             Relu, Dense(1))
opt_init, opt_update, get_params = adam(step_size=0.001)

def loss_fn(params, X, y):
    y_pred = apply_fun(params, X)
    mse = jnp.mean((y_pred - y) ** 2)
    return mse

@jax.jit
def update(i, opt_state, X, y):
    params = get_params(opt_state)
    loss, grads = jax.value_and_grad(loss_fn)(params, X, y)
    # clipped_grads = clip_grads(grads, max_norm=30.0)
    clipped_grads = tree_map(lambda w: jnp.clip(w, -1.0, 1.0), grads)
    # clipped_grads = grads
    opt_state = opt_update(i, clipped_grads, opt_state)
    return loss, opt_state, grads, clipped_grads

def train(rng):
    input_shape = (500, 500)
    batch_size = 256
    X = jax.random.normal(rng, shape=(batch_size,) + input_shape)
    y = jnp.array([jnp.mean(x ** 2) for x in X])

    _, params = init_fun(rng, (-1,) + input_shape)
    opt_state = opt_init(params)

    num_iters = 100
    print_every = 10
    for it in range(num_iters):
        loss, opt_state, grads, clipped_grads = update(0, opt_state, X, y)
        if it % print_every == 0:
            print("loss: {:.16f}\tgr_n: {:.16f}\tcl_g_n: {:.16f}".format(
                        loss, l2_norm(grads), l2_norm(clipped_grads)))

if __name__ == "__main__":
    seed = 0
    rng = jax.random.PRNGKey(seed)
    print("WITH JIT")
    train(rng)