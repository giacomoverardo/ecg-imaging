from tensorflow.keras.layers import Layer
import tensorflow as tf

class Sampling(Layer):
    """Source: https://keras.io/examples/generative/vae/ """
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

if __name__ == '__main__':
    pass