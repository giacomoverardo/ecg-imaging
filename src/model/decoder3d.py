"""Module providing a tensorflow model for decoding of
 an ECG imaging task
"""

from keras.layers import (
    InputLayer,
    Dense,
    Reshape,
    Conv3DTranspose,
    Conv3D,
    Concatenate,
    AvgPool3D,
    LeakyReLU,
    Conv2DTranspose,
)
from keras.models import Sequential, Model
from keras.regularizers import L1L2
import tensorflow as tf


class Decoder3D(Model):
    """Decoder of an ECGI model

    Args:
        Model (_type_): Base TensorFlow model
    """

    def __init__(self, sequence_length, num_3d_filters, l1_coeff, l2_coeff):
        super(Decoder3D, self).__init__()
        # Input layers for signal
        self.reshape_sig_in = Reshape((20, 13, sequence_length))
        self.conv_in_sig1 = Conv2DTranspose(
            filters=200,
            kernel_size=(3, 3),
            strides=(1, 2),
            padding="same",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )
        self.conv_in_sig2 = Conv2DTranspose(
            filters=200,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding="same",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )
        self.conv_in_sig3 = Conv2DTranspose(
            filters=120,
            kernel_size=(3, 3),
            strides=(3, 2),
            padding="same",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )
        # Concat layer to merge condition bspm and conduct
        self.concat_c = Concatenate()
        # # Concat layer to merge activation map and signal input
        # self.concat_in = Concatenate()
        # Decoder layers
        self.dec_dense_in = Dense(
            30 * 30 * 15,
            activation="tanh",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )
        self.dec_reshape_in = Reshape((30, 30, 15))

        # Block 1
        # Subsample c to 30x30x15
        self.dec_pool1 = AvgPool3D(pool_size=(4, 4, 16))
        # Main branch
        self.dec_concat1 = Concatenate()
        self.dec_conv_transpose1 = Conv3DTranspose(
            num_3d_filters,
            kernel_size=3,
            strides=(2, 2, 1),
            activation="relu",
            padding="same",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )
        self.dec_conv1 = Conv3D(
            num_3d_filters,
            kernel_size=3,
            strides=1,
            activation="relu",
            padding="same",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )
        # Block 2
        # Subsample c to 60x60x30
        self.dec_pool2 = AvgPool3D(pool_size=(2, 2, 8))
        # Main branch
        self.dec_concat2 = Concatenate()
        self.dec_conv_transpose2 = Conv3DTranspose(
            num_3d_filters,
            kernel_size=3,
            strides=(2, 2, 2),
            activation="relu",
            padding="same",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )
        self.dec_conv2 = Conv3D(
            num_3d_filters,
            kernel_size=3,
            strides=1,
            activation="relu",
            padding="same",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )
        # Block 3
        # Subsample c to 120x120x60
        self.dec_pool3 = AvgPool3D(pool_size=(1, 1, 4))
        # Main branch
        self.dec_concat3 = Concatenate()
        self.dec_conv_transpose3 = Conv3DTranspose(
            num_3d_filters,
            kernel_size=3,
            strides=(1, 1, 2),
            activation=LeakyReLU(),
            padding="same",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )
        self.dec_conv3 = Conv3D(
            1,
            kernel_size=3,
            strides=1,
            activation="sigmoid",
            padding="same",
            kernel_regularizer=L1L2(l1=l1_coeff, l2=l2_coeff),
        )

    def get_condition_bspm(self, x_s: tf.Tensor) -> tf.Tensor:
        """Compute the conditioning vector from the bspm signal

        Args:
            x_s (tf.Tensor): Bspm signal

        Returns:
            tf.Tensor: Conditioning vector
        """
        # Reshape x_s and take only first 200 ms
        x_s = self.reshape_sig_in(x_s)[..., :200]
        # Apply padding
        paddings = [[0, 0], [0, 0], [0, 2], [0, 0]]
        x_s = tf.pad(x_s, paddings)
        # Apply 2 convolutional layers to signal
        x_s = self.conv_in_sig1(x_s)
        x_s = self.conv_in_sig2(x_s)
        c = self.conv_in_sig3(x_s)
        return c

    def decoding_step(self, z: tf.Tensor, c: tf.Tensor) -> tf.Tensor:
        """Apply decoder to reconstruct cardiac activation map
        from latent space

        Args:
            z (tf.Tensor): Latent variable
            c (tf.Tensor): Conditioning vector

        Returns:
            tf.Tensor: Cardiac activation map
        """
        c = tf.expand_dims(c, axis=-1)
        # Apply decoder
        x = self.dec_dense_in(z)
        x = self.dec_reshape_in(x)
        # Apply decoder block 1
        # Subsample conditional variable c
        c_sub_1 = tf.squeeze(self.dec_pool1(c), axis=-1)
        # Main branch
        x = self.dec_concat1((x, c_sub_1))
        x = tf.expand_dims(x, axis=-1)
        x = self.dec_conv_transpose1(x)
        x = self.dec_conv1(x)
        # Apply decoder block 2
        # Subsample conditional variable c
        c_sub_2 = self.dec_pool2(c)
        # Main branch
        x = self.dec_concat2((x, c_sub_2))
        x = self.dec_conv_transpose2(x)
        x = self.dec_conv2(x)
        # Apply decoder block 3
        # Subsample conditional variable c
        c_sub_3 = self.dec_pool3(c)
        # Main branch
        x = self.dec_concat3((x, c_sub_3))
        x = self.dec_conv_transpose3(x)
        x = self.dec_conv3(x)
        x = tf.squeeze(x, axis=-1)
        return x

    def call(self, dec_input, conduct, signal, training):
        c_bspm = self.get_condition_bspm(signal)
        c = self.concat_c((c_bspm, conduct))
        return self.decoding_step(dec_input, c)

if __name__ == "__main__":
    pass
