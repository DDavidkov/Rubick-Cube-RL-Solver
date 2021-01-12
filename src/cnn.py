import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.stax import Conv, Relu, Flatten, Dense, serial
from jax.nn.initializers import glorot_normal, normal
from jax.lax import conv_general_dilated
from jax.nn import relu


def conv_net():
    out_dim = 1
    dim_nums = ("NHWC", "HWIO", "NHWC")

    # Primary convolutional layer.
    conv_channels = 8
    conv_init, conv_apply = Conv(out_chan=conv_channels, filter_shape=(3,3),
                                 strides=(1,3), padding=[(0,0), (0,0)])
    # Group all possible pairs.
    pair_channels, filter_shape = 64, (1, 2)

    # Forward pass.
    serial_init, serial_apply = serial(Flatten, Dense(512), Relu, Dense(out_dim))

    def init_fun(rng, input_shape):
        rng, conv_rng, serial_rng = jax.random.split(rng, num=3)

        # Primary convolutional layer.
        conv_shape, conv_params = conv_init(conv_rng, (-1,) + input_shape)

        # Grouping all possible pairs.
        kernel_shape = [filter_shape[0], filter_shape[1], conv_channels, pair_channels]
        bias_shape = [1, 1, 1, pair_channels]
        W_init = glorot_normal(in_axis=2, out_axis=3)
        b_init = normal(1e-6)
        k1, k2 = jax.random.split(rng)
        W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape)
        pair_shape = conv_shape[:2] + (15,) + (pair_channels,)
        pair_params = (W, b)

        # Forward pass.
        serial_shape, serial_params = serial_init(serial_rng, pair_shape)
        params = [conv_params, pair_params, serial_params]
        return serial_shape, params

    def apply_fun(params, inputs):
        conv_params, pair_params, serial_params = params

        # Apply the primary convolutional layer.
        conv_out = conv_apply(conv_params, inputs)
        conv_out = relu(conv_out)

        # Group all possible pairs.
        W, b = pair_params
        stride, pad = (1,1), ((0,0),(0,0))
        pair_1 = conv_general_dilated(conv_out, W, stride, pad, (1,1), (1,1), dim_nums) + b
        pair_2 = conv_general_dilated(conv_out, W, stride, pad, (1,1), (1,2), dim_nums) + b
        pair_3 = conv_general_dilated(conv_out, W, stride, pad, (1,1), (1,3), dim_nums) + b
        pair_4 = conv_general_dilated(conv_out, W, stride, pad, (1,1), (1,4), dim_nums) + b
        pair_5 = conv_general_dilated(conv_out, W, stride, pad, (1,1), (1,5), dim_nums) + b
        pair_out = jnp.dstack([pair_1, pair_2, pair_3, pair_4, pair_5])
        pair_out = relu(pair_out)

        # Forward pass.
        out = serial_apply(serial_params, pair_out)
        return out

    return init_fun, apply_fun

#

# if __name__ == "__main__":
#     rng = jax.random.PRNGKey(0)
#     input_shape = (3, 18, 1)
#     init_fun, apply_fun = conv_net()
#     out_shape, params = init_fun(rng, input_shape)
#     # print("PARAMS:")
#     # print(params, "\n\n")

#     inputs = np.array([[i ** 1.0] for i in range(3*18)], dtype=np.float32).reshape(1, 3, 18, 1)

#     # print("inputs:\n", inputs)
#     vals = apply_fun(params, inputs)