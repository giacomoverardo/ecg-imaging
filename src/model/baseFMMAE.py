"""Module providing utils and neural network models
to handle and employ FMM parameters 
"""

import tensorflow as tf
import numpy as np
from keras.layers import Dense, Layer
from keras.activations import softplus
from typing import List, Tuple, Dict
from src.utils.nn import bounded_output
from src.utils.fmm import generate_wave_tf
from src.utils.metrics import masked_mse
from src.utils.math import cos_sin_vector_to_angle
from src.utils.fmm import (
    get_A_indexes_circular,
    get_alpha_indexes_circular,
    get_beta_indexes_circular,
    get_omega_indexes_circular,
    get_M_indexes_circular,
    get_fmm_num_parameters_circular,
    get_wave_names,
    convert_to_linear
)

start = tf.constant(0, dtype=tf.float32)
stop = tf.constant(2 * np.pi, dtype=tf.float32)
t = tf.linspace(start, stop, 450)


@tf.function
def fmm_wave_tf(params: tf.Tensor) -> tf.Tensor:
    """Generates an FMM wave based on the input FMM parameters

    Args:
        params (tf.Tensor): Tensors including FMM parameters A, alpha, beta, omega

    Returns:
        tf.Tensor: Output wave
    """
    # A, alpha, beta, omega = params
    A = params[0, ...]  # Assumes first dimension holds parameters
    alpha = params[1, ...]
    beta = params[2, ...]
    omega = params[3, ...]
    phase = beta + 2 * tf.math.atan(omega * tf.math.tan((t - alpha) / 2))
    # Wave: A cos (ϕ (t; α, β, ω))
    wave = A * tf.math.cos(phase)
    return wave


class FMM_head(Layer):
    def __init__(
        self,
        num_leads: int,
        num_waves: int,
        seq_len: int,
        max_omega: float,
        batch_size: int,
        trainable: bool = True,
        name: str = None,
        dtype=None,
        dynamic: bool = False,
        **kwargs
    ):
        super().__init__(trainable, name, dtype, dynamic)
        self.n = num_leads
        self.num_waves = num_waves
        self.num_parameters, self.num_parameters_per_wave = (
            get_fmm_num_parameters_circular(
                num_leads=num_leads, num_waves=self.num_waves
            )
        )
        self.dense_layers = [
            Dense(num_nodes, activation=act)
            for num_nodes, act in zip([2048, self.num_parameters], ["tanh", "linear"])
        ]
        self.softplus = softplus
        self.seq_len = seq_len
        self.max_omega = max_omega
        self.kwargs = kwargs
        self.wave_names = get_wave_names(num_waves)
        self.batch_size = batch_size

    def get_A(self, x, wave_index):
        # Get alpha FMM parameter from tensor x and corresponding wave index
        # Wave_index is usually the index in {P, Q, R, S, T},
        # but it may change depending on the number of total waves
        start_index, end_index = get_A_indexes_circular(
            wave_index=wave_index, num_leads=self.n, num_waves=self.num_waves
        )
        return x[:, start_index:end_index]

    def get_alpha(self, x, wave_index):
        # Get alpha FMM parameter from tensor x and corresponding wave index
        start_index, end_index = get_alpha_indexes_circular(
            wave_index=wave_index, num_leads=self.n, num_waves=self.num_waves
        )
        return x[:, start_index:end_index]

    def get_all_alpha_linear(self, x):
        # Get linear alpha FMM parameter from tensor x for all the waves
        linear_alpha = [] #tf.zeros((x.shape[0],self.num_waves),dtype=x.dtype)
        for w_i in range(self.num_waves):
            alpha_i_sin_cos = self.get_alpha(x,w_i) * 2 - 1
            alpha_i = cos_sin_vector_to_angle(alpha_i_sin_cos)
            # alpha_i_linear = tf.map_fn(convert_to_linear,alpha_i)
            alpha_i_linear_sorted = tf.sort(alpha_i,axis=0)
            linear_alpha.append(alpha_i_linear_sorted)
        linear_alpha_stacked = tf.concat(linear_alpha,axis=1)
        return linear_alpha_stacked

    def get_beta(self, x, wave_index):
        # Get beta FMM parameter from tensor x and corresponding wave index
        start_index, end_index = get_beta_indexes_circular(
            wave_index=wave_index, num_leads=self.n, num_waves=self.num_waves
        )
        return x[:, start_index:end_index]

    def get_omega(self, x, wave_index):
        # Get omega FMM parameter from tensor x and corresponding wave index
        start_index, end_index = get_omega_indexes_circular(
            wave_index=wave_index, num_leads=self.n, num_waves=self.num_waves
        )
        return x[:, start_index:end_index]

    def get_M(self, x):
        # Get M FMM parameter from tensor x
        start_index, end_index = get_M_indexes_circular(
            wave_index=None, num_leads=self.n, num_waves=self.num_waves
        )
        # Return 0 for this case since we have zero offset in all the signals
        return tf.zeros_like(x[:, start_index:end_index])
        # return x[:, start_index:end_index]

    def split_parameters(self, x) -> Dict:
        parameters_dict = {}
        m = self.get_M(x)
        for i, w in enumerate(self.wave_names):
            a = self.get_A(x, wave_index=i)
            alpha = self.get_alpha(x, wave_index=i) * 2 - 1
            alpha = cos_sin_vector_to_angle(alpha)
            beta = self.get_beta(x, wave_index=i) * 2 - 1
            beta = cos_sin_vector_to_angle(beta)
            omega = self.get_omega(x, wave_index=i)
            parameters_dict[w] = {
                "A": a,
                "alpha": alpha,
                "beta": beta,
                "omega": omega,
                "M": m,
            }
        return parameters_dict

    def scale_parameters(self, parameters_array, up_alpha, up_beta, up_omega):
        return NotImplementedError()

    def bound_parameters(self, parameters_array):
        # return parameters_array
        x = parameters_array
        bounded_parameters_array = []
        for i, w in enumerate(self.wave_names):
            a = self.get_A(x, wave_index=i)
            a = self.softplus(a)
            alpha = self.get_alpha(x, wave_index=i)
            alpha = bounded_output(alpha, lower=0.0, upper=1.0)
            # alpha = bounded_output(alpha,lower=0,upper=self.upper_limit_alpha_beta)
            beta = self.get_beta(x, wave_index=i)
            beta = bounded_output(beta, lower=0.0, upper=1.0)
            omega = self.get_omega(x, wave_index=i)
            omega = bounded_output(omega, lower=0, upper=self.max_omega)
            # omega = bounded_output(omega,lower=0,upper=1.0)
            bounded_parameters_array.append(a)
            bounded_parameters_array.append(alpha)
            bounded_parameters_array.append(beta)
            bounded_parameters_array.append(omega)
        m = self.get_M(x)
        bounded_parameters_array.append(m)
        bounded_parameters_array = tf.concat(bounded_parameters_array, axis=1)
        # bounded_parameters_array = tf.squeeze(tf.stack(bounded_parameters_array,axis=1))
        return bounded_parameters_array

    def get_wave(self, parameters_dict, wave_name, lead, seq_len):
        return generate_wave_tf(
            parameters_dict=parameters_dict,
            wave_name=wave_name,
            lead=lead,
            seq_len=seq_len,
            split_ecg=False,
        )

    def split_angular_parameters(self, parameters_array_bounded):
        x = parameters_array_bounded
        parameters_array = []
        for i, w in enumerate(self.wave_names):
            a = self.get_A(x, wave_index=i)
            alpha = self.get_alpha(x, wave_index=i) * 2 - 1
            alpha = cos_sin_vector_to_angle(alpha)
            beta = self.get_beta(x, wave_index=i) * 2 - 1
            beta = cos_sin_vector_to_angle(beta)
            omega = self.get_omega(x, wave_index=i)
            parameters_array.append(a)
            parameters_array.append(alpha)
            parameters_array.append(beta)
            parameters_array.append(omega)
        m = self.get_M(x)
        parameters_array.append(m)
        # parameters_array = tf.squeeze(tf.stack(parameters_array,axis=1))
        parameters_array = tf.squeeze(tf.concat(parameters_array, axis=1))
        return parameters_array

    # def call(self, inputs, *args, **kwargs):
    #     return self.__call__(inputs)

    def call(
        self,
        x,
        x_len=None,
        return_parameters_dict=False,
        return_parameters_array: bool = False,
        return_parameters_array_bounded: bool = False,
    ):
        result = {}
        for layer in self.dense_layers:
            x = layer(x)
        parameters_array_bounded = self.bound_parameters(x)
        parameters_array = self.split_angular_parameters(parameters_array_bounded)
        parameters_dict = self.split_parameters(parameters_array_bounded)
        if return_parameters_dict:
            result["parameters_dict"] = parameters_dict
        if return_parameters_array_bounded:
            result["parameters_array_bounded"] = parameters_array_bounded
        # def fmm_wave_tf(A,alpha,beta,omega):

        # leads = []
        tf_leads = tf.zeros((self.batch_size, self.seq_len, self.n), dtype=tf.float32)
        for j, w in enumerate(self.wave_names):
            arg_list = []
            for i in range(self.n):
                a = tf.reshape(parameters_dict[w]["A"][:, i], (-1, 1))
                alpha = tf.reshape(parameters_dict[w]["alpha"], (-1, 1))
                beta = tf.reshape(parameters_dict[w]["beta"][:, i], (-1, 1))
                omega = tf.reshape(parameters_dict[w]["omega"], (-1, 1))
                arg_list.append([a, alpha, beta, omega])
            # waves = tf.map_fn(fmm_wave_tf, tf.stack(arg_list,axis=0),parallel_iterations=50)
            waves = tf.vectorized_map(fmm_wave_tf, tf.stack(arg_list, axis=0))
            waves = tf.experimental.numpy.swapaxes(waves, 0, 1)
            waves = tf.experimental.numpy.swapaxes(waves, 1, 2)
            tf_leads += waves
        result["output"] = tf_leads
        if return_parameters_array:
            result["parameters_array"] = parameters_array
        return result


class Base_FMM_Model(tf.keras.Model):
    def __init__(
        self,
        num_leads,
        num_waves,
        seq_len,
        max_omega,
        batch_size,
        reconstruction_loss_weight,
        coefficient_loss_weight,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder = None
        self.global_avg_pooling = None
        self.seq_len = seq_len
        # self.encoder_input_type = "tensor" # or "dict" if we feed the whole input dictionary
        self.fmm_head = FMM_head(
            num_leads=num_leads,
            num_waves=num_waves,
            seq_len=seq_len,
            max_omega=max_omega,
            batch_size=batch_size,
        )
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.coefficient_loss_weight = coefficient_loss_weight
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # self.coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.test_reconstruction_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        # self.test_coefficient_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.test_total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            # self.coefficient_loss_tracker,
            self.test_total_loss_tracker,
            self.test_reconstruction_loss_tracker,
            # self.test_coefficient_loss_tracker
        ]

    def get_encoded_parameters(self, inputs):
        x = inputs["signal"]
        x = self.encoder(x)
        x = self.global_avg_pooling(x)
        for layer in self.fmm_head.dense_layers:
            x = layer(x)
        parameters_dict = self.fmm_head.split_parameters(x)
        return parameters_dict

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss_dict = self.compute_loss(self(inputs=data, training=False))
            total_loss_mean = tf.reduce_mean(loss_dict["loss"])
        grads = tape.gradient(total_loss_mean, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss_dict["loss"])
        self.reconstruction_loss_tracker.update_state(loss_dict["rec_loss"])
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
        }

    def test_step(self, data):
        loss_dict = self.compute_loss(self(inputs=data, training=False))
        self.test_total_loss_tracker.update_state(loss_dict["loss"])
        self.test_reconstruction_loss_tracker.update_state(loss_dict["rec_loss"])
        return {
            "loss": self.test_total_loss_tracker.result(),
            "rec_loss": self.test_reconstruction_loss_tracker.result(),
        }

    def call(self, inputs, training):
        x = inputs["signal"]
        x = tf.experimental.numpy.swapaxes(x, 1, 2)
        x = self.encoder(
            inputs=x, training=training
        )  # (batch_size, context_len, d_model)
        if self.global_avg_pooling is not None:
            x = self.global_avg_pooling(x)
        fmm_head_output_dict = self.fmm_head(
            x,
            x_len=self.seq_len,
            return_parameters_array=True,
            return_parameters_array_bounded=True,
        )
        x = fmm_head_output_dict["output"]
        # x = tf.zeros_like(inputs["signal"])
        return {
            "signal": inputs["signal"],
            "predicted_signal": x,
            # "fmm_coefficients":inputs["coefficients"],
            # "fmm_coefficients_ang":inputs["coefficients_ang"],
            "predicted_fmm_coefficients": fmm_head_output_dict["parameters_array"],
            "predicted_fmm_coefficients_bounded": fmm_head_output_dict[
                "parameters_array_bounded"
            ],
        }

    def compute_loss(self, inputs):
        predicted_data = inputs["predicted_signal"]
        data = inputs["signal"]
        reconstruction_loss = masked_mse(
            tf.experimental.numpy.swapaxes(data, 1, 2), predicted_data
        )
        total_loss = reconstruction_loss
        return {"loss": total_loss, "rec_loss": reconstruction_loss}


if __name__ == "__main__":
    pass
