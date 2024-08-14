"""Module providing a tensorflow model for ECG imaging based on 
    @article{ECGIwithCVAEsupplement,
    Tania Bacoyannis et al. “Deep learning formulation of electrocardiographic
    imaging integrating image and signal information with data-driven regu-
    larization”. In: EP Europace 23.Supplement1 (Mar. 2021), pp. i55–i62.
    issn: 1099-5129. doi: 10.1093/europace/euaa391. 
    url: https://doi.org/10.1093/europace/euaa391.
"""

import tensorflow as tf
from keras.layers import (
    Dense,
    Flatten,
    Reshape,
    Conv3D,
    Concatenate,
    Conv3DTranspose,
    LeakyReLU,
    AvgPool3D,
    Conv2DTranspose,
)
from keras.metrics import Mean
from src.model.base import BaseModel
from src.model.vae import Sampling
from src.utils.metrics import inverted_mask_contrast_mse, inverted_mask_contrast_mae


class VaeInria(BaseModel):
    """VAE model for ECG Imaging

    Args:
        BaseModel (tensorflow.Model): Base model class to perform ECGI
    """

    def __init__(
        self,
        sequence_length,
        num_features,
        output_size,
        alpha,
        beta,
        latent_size,
        **kwargs
    ):
        super().__init__(**kwargs)
        # Train trackers
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(name="rec_loss")
        self.reconstruction_loss_gaussian_tracker = Mean(name="rec_loss_gaus")
        self.kl_loss_tracker = Mean(name="kl_loss")
        self.reconstruction_mae_acti_loss_tracker = Mean(name="rec_loss_mae")
        # Test trackers
        self.test_total_loss_tracker = Mean(name="test_total_loss")
        self.test_reconstruction_loss_tracker = Mean(name="test_rec_loss")
        self.test_reconstruction_loss_gaussian_tracker = Mean(name="test_rec_loss_gaus")
        self.test_kl_loss_tracker = Mean(name="test_kl_loss")
        self.test_reconstruction_mae_acti_loss_tracker = Mean(name="test_rec_loss_mae")
        # Store input parameters
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.output_size = output_size
        self.alpha = alpha
        self.beta = beta
        self.latent_size = latent_size
        num_3d_filters = 4
        # Input layers for signal
        self.reshape_sig_in = Reshape((20, 13, sequence_length))
        self.conv_in_sig1 = Conv2DTranspose(
            filters=200, kernel_size=(3, 3), strides=(1, 2), padding="same"
        )
        self.conv_in_sig2 = Conv2DTranspose(
            filters=200, kernel_size=(3, 3), strides=(2, 2), padding="same"
        )
        self.conv_in_sig3 = Conv2DTranspose(
            filters=120, kernel_size=(3, 3), strides=(3, 2), padding="same"
        )
        # Concat layer to merge condition bspm and conduct
        self.concat_c = Concatenate()
        # Concat layer to merge activation map and signal input
        self.concat_in = Concatenate()
        # Encoder layers
        self.enc_conv1 = Conv3D(
            filters=num_3d_filters, kernel_size=3, strides=2, activation="tanh", input_shape=(None, 120, 120, 360, 1)
        )
        self.enc_conv2 = Conv3D(
            filters=num_3d_filters, kernel_size=3, strides=2, activation="tanh"
        )
        self.enc_flatten = Flatten()
        self.enc_dense_out = Dense(latent_size, activation="tanh")
        # Latent space and sampling
        self.enc_dense_mu = Dense(latent_size)
        self.enc_dense_sigma = Dense(latent_size)
        self.sampling = Sampling()
        # Decoder layers
        self.dec_dense_in = Dense(30 * 30 * 15, activation="tanh")
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
        )
        self.dec_conv1 = Conv3D(
            num_3d_filters, kernel_size=3, strides=1, activation="relu", padding="same"
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
        )
        self.dec_conv2 = Conv3D(
            num_3d_filters, kernel_size=3, strides=1, activation="relu", padding="same"
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
        )
        self.dec_conv3 = Conv3D(
            1, kernel_size=3, strides=1, activation="sigmoid", padding="same"
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
        # c = tf.expand_dims(c, axis=-1)
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
        # c_sub_1 = self.dec_pool1(c)
        # Main branch
        x = self.dec_concat1((x, c_sub_1))
        x = tf.expand_dims(x, axis=-1)
        x = self.dec_conv_transpose1(x)
        x = self.dec_conv1(x)
        # Apply decoder block 2
        # Subsample conditional variable c
        # c_sub_2 = tf.squeeze(self.dec_pool2(c), axis=-1)
        c_sub_2 = self.dec_pool2(c)
        # Main branch
        # x = tf.squeeze(x,axis=-1)
        x = self.dec_concat2((x, c_sub_2))
        # x = tf.expand_dims(x, axis=-1)
        x = self.dec_conv_transpose2(x)
        x = self.dec_conv2(x)
        # Apply decoder block 3
        # Subsample conditional variable c
        # c_sub_3 = tf.squeeze(self.dec_pool3(c), axis=-1)
        c_sub_3 = self.dec_pool3(c)
        # Main branch
        # x = tf.squeeze(x,axis=-1)
        x = self.dec_concat3((x, c_sub_3))
        # x = tf.expand_dims(x, axis=-1)
        x = self.dec_conv_transpose3(x)
        x = self.dec_conv3(x)
        x = tf.squeeze(x, axis=-1)
        return x

    def call(self, inputs, training):
        x_a = inputs["acti_map"]
        # x_a = tf.sparse.reshape(x_a, [-1, 120, 120, 120])  # Ensure x_a shape is defined
        c_conduct = inputs["conduct"]
        # c_conduct = tf.sparse.reshape(c_conduct, [-1, 120, 120, 120])  # Ensure c_conduct shape is defined
        c_bspm = self.get_condition_bspm(inputs["signal"])
        c_bspm = tf.reshape(c_bspm, [-1, 120, 120, 120])  # Assuming c_bspm is a dense tensor
        c = self.concat_c((c_bspm, c_conduct))
        c = tf.reshape(c, [-1, 120, 120, 240])  # Assuming the result of concat_c is a dense tensor
        x = self.concat_in((x_a, c))
        x = tf.reshape(x, [-1, 120, 120, 360])  # Ensure x shape is defined
        x = tf.expand_dims(x, axis=-1)
        x = self.enc_conv1(x)
        x = self.enc_conv2(x)
        x = self.enc_flatten(x)
        x = tf.reshape(x, [-1, 299396])  # hard code shape since set_shape is not available in tf 2.10 as in leonardo HPC
        mu = self.enc_dense_mu(x)
        sigma = self.enc_dense_sigma(x)
        # Sample from latent space
        z = self.sampling((mu, sigma))
        x = self.decoding_step(z, c)
        return {
            "signal": inputs["signal"],
            "acti_map": inputs["acti_map"],
            "conduct": inputs["conduct"],
            "mask": inputs["mask"],
            "predicted_acti_map": x,
            "z_mean": mu,
            "z_log_var": sigma,
            "c": c,
        }

    def compute_loss(self, inputs):
        # Compute reconstruction loss by drawing sample from the latent space
        # reconstruction_loss = super(VaeInria,self).compute_loss(inputs)["total_loss"]
        acti_map, predicted_acti_map, mask = (
            inputs["acti_map"],
            inputs["predicted_acti_map"],
            inputs["mask"],
        )
        reconstruction_loss = inverted_mask_contrast_mse(
            acti_map, predicted_acti_map, mask
        )
        rec_loss_mae = inverted_mask_contrast_mae(acti_map, predicted_acti_map, mask)
        # Compute kl loss
        z_mean, z_log_var = inputs["z_mean"], inputs["z_log_var"]
        kl_loss_vec = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss_vec, axis=1)
        # Compute reconstruction loss by drawing sample from a multivariate gaussian
        z_gaussian = self.sampling((tf.zeros_like(z_mean), tf.ones_like(z_log_var)))
        # Get predicted activation map from gaussian sample and condition vector
        predicted_acti_map_gaussian = self.decoding_step(z_gaussian, inputs["c"])
        # Compute reconstruction loss
        reconstruction_loss_gaussian = inverted_mask_contrast_mse(
            acti_map, predicted_acti_map_gaussian, mask
        )
        # Compute total loss
        total_loss = (
            self.alpha * (reconstruction_loss + reconstruction_loss_gaussian)
            + self.beta * kl_loss
        )
        return {
            "total_loss": total_loss,
            "rec_loss": reconstruction_loss,
            "rec_loss_gaus": reconstruction_loss_gaussian,
            "kl_loss": kl_loss,
            "rec_loss_mae":rec_loss_mae,
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.reconstruction_loss_gaussian_tracker,
            self.kl_loss_tracker,
            self.reconstruction_mae_acti_loss_tracker,
            self.test_total_loss_tracker,
            self.test_reconstruction_loss_tracker,
            self.test_reconstruction_loss_gaussian_tracker,
            self.test_kl_loss_tracker,
            self.test_reconstruction_mae_acti_loss_tracker
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss_dict = self.compute_loss(self(inputs=data, training=True))
            total_loss_mean = tf.reduce_mean(loss_dict["total_loss"])
        grads = tape.gradient(total_loss_mean, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.reconstruction_loss_tracker.update_state(loss_dict["rec_loss"])
        self.reconstruction_loss_gaussian_tracker.update_state(
            loss_dict["rec_loss_gaus"]
        )
        self.kl_loss_tracker.update_state(loss_dict["kl_loss"])
        self.total_loss_tracker.update_state(loss_dict["total_loss"])
        self.reconstruction_mae_acti_loss_tracker.update_state(loss_dict["rec_loss_mae"])
        return {
            "loss": self.total_loss_tracker.result(),
            "rec_loss": self.reconstruction_loss_tracker.result(),
            "rec_loss_gaus": self.reconstruction_loss_gaussian_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "rec_loss_mae": self.reconstruction_mae_acti_loss_tracker.result(),
        }

    def test_step(self, data):
        loss_dict = self.compute_loss(self(inputs=data, training=False))
        self.test_reconstruction_loss_tracker.update_state(loss_dict["rec_loss"])
        self.test_reconstruction_loss_gaussian_tracker.update_state(
            loss_dict["rec_loss_gaus"]
        )
        self.test_total_loss_tracker.update_state(loss_dict["total_loss"])
        self.test_reconstruction_mae_acti_loss_tracker.update_state(loss_dict["rec_loss_mae"])
        return {
            "loss": self.test_total_loss_tracker.result(),
            "rec_loss": self.test_reconstruction_loss_tracker.result(),
            "rec_loss_gaus": self.test_reconstruction_loss_gaussian_tracker.result(),
            "kl_loss": self.test_kl_loss_tracker.result(),
            "rec_loss_mae": self.test_reconstruction_mae_acti_loss_tracker.result(),
        }


if __name__ == "__main__":
    pass
