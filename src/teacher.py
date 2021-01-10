import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from functools import partial

from gather_data import generate_episodes, make_targets, batch_generator


class Teacher:
    """ 
    A Teacher object encapsulates all the logic necessary for training
    reinforcement learning agents lerning to solve the Rubik's cube.
    """
    #--------- teacher initializer --------#
    def __init__(self, env, model_fn, **kwargs):
        """
        Initialize an instance of the Teacher class.

        Required arguments:
        @param env (Cube object): A Cube object representing the environment.
        @param model_fn: A model function.
        @param out_dim (Tuple(int)): A tuple of intigers giving the shape of the output of the model function.

        Optional arguments:
        @param step_size (float): A scalar giving the step size.
        @param decay_rate (float): A scalar for decaying the step size.
        @param reg (float): A scalar giving the L2-regularization strength.
        @param batch_size (int): Size of minibatches used to compute loss and gradient during training.
        @param num_epochs (int): The number of epochs to run for during training.
        @param num_iterations (int): The number of iterations through the generated episodes.
        @param num_samples (int): The number of times the dataset is reused.
        @param print_every (int): An integer. Training progress will be printed every `print_every` iterations.
        @param episodes (int):
        @param k_max(int)
        @param clip_norm (float): A scalar for gradient clipping.
        @param patience (int):
        @param max_num_trial (int): number of trials before termination.
        @param verbose (bool): If set to false then no output will be printed during training.
        @param params_save_path (str): File path to save the model parameters.
        """
        # Unpack keyword arguments.
        self.step_size = kwargs.pop("step_size", 1e-3)
        self.decay_rate = kwargs.pop("decay_rate", 1.0)         # decay_rate = 1.0 ==> lr_decay = 1/(1+decay_rate) = 0.5
        self.reg = kwargs.pop("reg", 0.0)
        self.batch_size = kwargs.pop("batch_size", 128)
        self.num_epochs = kwargs.pop("num_epochs", 10)
        self.num_iterations = kwargs.pop("num_iterations", 101)
        self.num_samples = kwargs.pop("num_samples", 1000)
        self.print_every = kwargs.pop("print_every", 10)
        self.episodes = kwargs.pop("episodes", 10)
        self.k_max = kwargs.pop("k_max", 10)
        self.clip_norm = kwargs.pop("clip_norm", np.inf)
        self.patience = kwargs.pop("patience", np.inf)
        self.max_num_trial = kwargs.pop("max_num_trial", np.inf)
        self.verbose = kwargs.pop("verbose", True)
        self.params_save_path = kwargs.pop("params_save_path", None)

        # Throw an error if there are extra keyword arguments.
        if len(kwargs) > 0:
            extra = ', '.join("'%s'" % k for k in list(kwargs.keys()))
            raise ValueError("Unrecognized arguments %s" % extra)

        # Environment.
        self.env = env

        # Define value-approximation function.
        self.init_fun, self.apply_fun = model_fn()

        # Define optimizer.
        self.step_fn = optimizers.inverse_time_decay(step_size=self.step_size,
                                                     decay_rate=self.decay_rate,
                                                     decay_steps=1)
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=self.step_fn)

    #----------- public methods -----------#
    def train(self, rng, input_shape=(3, 18, 6)):
        """
        Train the model function by generating simulations of random-play.
        On every epoch generate a new simulation and run multiple iterations.
        On every iteration evaluate the targets using the most recent model parameters
        and run multiple times through the dataset.
        At the end of every epoch check the performance and store the best performing params.
        If the performance drops then decay the step size parameter.

        @param rng (PRNGKey): A pseudo-random number generator.
        @param input_shape (Tuple(int)): A tuple of intigers giving the shape of the inputs.
        @returns params (pytree): The best model parameters obatained during training.
        """
        self._reset()

        # Initialize model parameters and optimizer state.
        rng, init_rng = jax.random.split(rng)
        _, params = self.init_fun(init_rng, input_shape)
        opt_state = self.opt_init(params)

        # Begin training.
        # step_count = patience = num_trial = 0
        for e in range(self.num_epochs):
            tic = time.time()

            # Generate data from random-play using the environment.
            rng, sub_rng = jax.random.split(rng)
            states, w, children, rewards = generate_episodes(sub_rng, self.env, self.episodes, self.k_max)

            # # Store the optimizer state that generates the results.
            # prev_state = opt_state

            # Train the model on the generated data. Periodically recompute the target values.
            epoch_mean_loss = 0.0
            for it in range(self.num_iterations):
                tic_it = time.time()

                # Make targets for the generated episodes using the most recent params and build a batch generator.
                params = self.get_params(opt_state)
                tgt_vals, tgt_acts = make_targets(children, rewards, params, self.apply_fun)
                data = {"X" : states, "y" : (tgt_vals, tgt_acts), "w" : w}
                rng, sub_rng = jax.random.split(rng)
                train_batches = batch_generator(sub_rng, data, self.batch_size)

                # Run through the dataset and update model params.
                total_loss = 0.0
                for i in range(self.num_samples):
                    batch = next(train_batches)
                    loss, opt_state = self.update(e, opt_state, batch)
                    total_loss += loss

                # Book-keeping.
                iter_mean_loss = total_loss / self.num_samples
                epoch_mean_loss = (it * epoch_mean_loss + iter_mean_loss) / (it + 1)
                self.loss_history.append(iter_mean_loss)

                toc_it = time.time()
                if it % self.print_every == 0 and self.verbose:
                    print("\t(Iteration({}/{}) took {:.3f} seconds) iter_mean_loss = {:.3f}".format(
                                                                    it + 1, self.num_iterations, (toc_it-tic_it), iter_mean_loss))

            # # If the model is performing better than it was on the previous epoch, then save the model parameters.
            # # If the model is performing worse than it was on the previous epoch, then increase the patience.
            # # if the patience reaches a limit, reload the previous parameters, decay the learning rate, and icrease trial count.
            # if epoch_mean_loss < 1.2 * self.best_loss:   #### 1.2 because training still takes place !
            #     self.best_loss = epoch_mean_loss
            #     self.best_state = prev_state
            #     if self.params_save_path is not None:
            #         jnp.save(self.params_save_path, self.get_params(self.best_state))
            #     patience = 0
            # else:
            #     print("Increasing patience.")
            #     patience += 1
            #     if patience >= self.patience:
            #         print("Loading the previous best model.")
            #         opt_state = self.best_state

            #         # Reset patience.
            #         patience = 0

            #         # Increase the trial count.
            #         num_trial += 1

            #         # Decay the step size by incrementing the step count of the optimizer.
            #         step_count += 1

            # # If the trial count reaches the maximum number of trials, stop the training.
            # if num_trial >= self.max_num_trial:
            #     print("Reached maximum number of trials!")
            #     break

            # Record the time needed for a single epoch.
            toc = time.time()

            # Printout results.
            if self.verbose:
                print("(Epoch ({}/{}) took {:.3f} seconds), epoch_mean_loss = {:.3f}".format(
                                                            e + 1, self.num_epochs, (toc - tic), epoch_mean_loss))

        return params

    #----------- private methods ----------#
    def _reset(self):
        self.loss_history = []
        self.best_loss = np.inf
        self.best_state = None

    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, batch):
        """ Return the total loss computed for a given batch. """
        X, (y_vals, y_acts), w = batch
        vhead_out, phead_out = self.apply_fun(params, X)
        # num_train = X.shape[0]
        mse_loss = np.sum(((vhead_out - y_vals) ** 2).squeeze() * w)# / np.sum(w)
        # cross_entropy_loss = - np.sum(phead_out[np.arange(num_train), y_acts.squeeze()] * w)# / np.sum(w)
        return mse_loss #+ cross_entropy_loss + l2_loss

    @partial(jax.jit, static_argnums=(0,))
    def update(self, i, opt_state, batch):
        """ Perform backpropagation and parameter update. """
        params = self.get_params(opt_state)
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch)
        # return loss, self.opt_update(i, optimizers.clip_grads(grads, self.clip_norm), opt_state)   # clip_grads produces different result when under jit !!!!!!
        return loss, self.opt_update(i, grads, opt_state)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(seed=0)

    from cube_model_naive import Cube
    from fcnn import fc_net

    teacher = Teacher(Cube, fc_net,
                    step_size=1e-3,
                    decay_rate=0.05,
                    reg=1e-4,
                    batch_size=256,
                    num_epochs=1,
                    num_iterations=26,
                    num_samples=100,
                    print_every=5,
                    episodes=2000,
                    k_max=10,
                    clip_norm=np.inf,
                    verbose=True,
                    params_save_path=None,
                    patience=2,
                    max_num_trial=15,
                    )

    # params = teacher.train(rng)
