"""Module providing a tensorflow model for ECG imaging based on
a GNN and FMM-based encoder
"""

from typing import List, Tuple, Union
import tensorflow as tf
from keras import layers
from keras.models import Model, Sequential
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Flatten,
    Conv1DTranspose,
    Dropout,
    InputLayer,
    AveragePooling1D,
    UpSampling1D,
    Add,
)
from keras.activations import tanh
from src.model.vae import Sampling


class CvaeBlock(Model):
    """CVAE model basic block from
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260612

    Args:
        Model (_type_): Base TensorFlow model
    """

    def __init__(
        self,
        kernel_size: int,
        num_filters: int = 1,
        dropout_rate: float = 0.1,
        add_avg_pool: bool = True,
        add_skip_connection: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.add_avg_pool = add_avg_pool
        self.batch_norm1 = BatchNormalization()
        self.add_skip_connection = add_skip_connection
        if add_skip_connection:
            self.convinput = Conv1D(
                filters=self.num_filters, kernel_size=1, padding="same"
            )
        self.conv1 = Conv1D(
            filters=self.num_filters, kernel_size=self.kernel_size, padding="same"
        )
        self.conv2 = Conv1D(
            filters=self.num_filters, kernel_size=self.kernel_size, padding="same"
        )
        self.avg_pool = AveragePooling1D(padding="same")
        self.batch_norm2 = BatchNormalization()
        self.batch_norm3 = BatchNormalization()
        self.dropout = Dropout(rate=self.dropout_rate)
        self.add = Add()
        self.activation = tanh

    # def call(self, inputs, training=None, mask=None):
    #     return self(inputs=inputs, training=training)

    def call(self, inputs, training=None, mask=None) -> tf.Tensor:
        x = inputs
        x = self.batch_norm1(inputs=x, training=training)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.batch_norm2(inputs=x, training=training)
        x = self.activation(x)
        x = self.conv2(x)
        if self.add_skip_connection:
            convinputs = self.convinput(inputs)
            convinputs = self.batch_norm3(inputs=convinputs, training=training)
            x = self.add([x, convinputs])
        if self.add_avg_pool:
            x = self.avg_pool(x)
        x = self.activation(x)
        if training:
            x = self.dropout(x)
        return x


class CvaeBlockDecoder(CvaeBlock):
    """CVAE model decoder block from
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260612

    Args:
        Model (_type_): CVAE basic block
    """

    def __init__(
        self,
        kernel_size: int,
        num_filters: int = 1,
        dropout_rate: float = 0.1,
        add_avg_pool: bool = True,
    ) -> None:
        super().__init__(
            kernel_size,
            num_filters,
            dropout_rate,
            add_avg_pool,
            add_skip_connection=True,
        )
        self.conv1 = Conv1DTranspose(self.num_filters, self.kernel_size, padding="same")
        self.conv2 = Conv1DTranspose(self.num_filters, self.kernel_size, padding="same")
        self.avg_pool = UpSampling1D()


class CvaeBlockEncoder(CvaeBlock):
    """CVAE model encoder block from
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260612

    Args:
        CvaeBlock (_type_): CVAE basic block
    """

    def __init__(
        self,
        kernel_size: int,
        num_filters: int = 1,
        dropout_rate: float = 0.1,
        add_avg_pool: bool = True,
    ) -> None:
        super().__init__(kernel_size, num_filters, dropout_rate, add_avg_pool)


class CvaeCoder(Model):
    """CVAE model general coder from
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260612

    Args:
        Model (_type_): Base TensorFlow model
    """

    def __init__(
        self,
        input_shape: Tuple,
        filter_size_list: List,  # = [19, 19, 19, 19, 19, 19, 9, 9, 9]
        num_filters_list: List,  # = [16, 16, 16, 32, 48, 64, 64, 80, 80, 80],
        add_avg_pool_list: List,  # = [1, 1, 1, 1, 1, 0, 0, 0, 0],
        sequential_part: Sequential = None,
    ) -> None:
        super().__init__()
        self.sequential_part = (
            Sequential() if sequential_part is None else sequential_part
        )
        if input_shape:
            self.sequential_part.add(InputLayer(input_shape=input_shape))
        for kernel_size, num_filters, add_pool in zip(
            filter_size_list, num_filters_list, add_avg_pool_list
        ):
            self.sequential_part.add(
                self.block_type(
                    kernel_size=kernel_size,
                    num_filters=num_filters,
                    add_avg_pool=add_pool,
                )
            )

    # def __call__(self, inputs, training=None, mask=None):
    #     return self.sequential_part(inputs=inputs, training=training)

    def get_sequential_output_shape(self, sample: tf.Tensor) -> tf.TensorShape:
        """Computes output shape of sequential part of the encoder given an input sample

        Args:
            sample (tf.Tensor): Input sample

        Returns:
            tf.TensorShape: Output shape of the sequential part
        """
        out = self.sequential_part(sample)
        return out.shape

    def call(self, inputs, training=None, mask=None):
        return self.sequential_part(inputs=inputs, training=training)


class CvaeEncoder(CvaeCoder):
    """CVAE model encoder from
    https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0260612

    Args:
        CvaeCoder (_type_): General CVAE coder
    """

    def __init__(
        self,
        sequence_length: int,
        num_features: int,
        latent_dim: int,
        filter_size_list: List,  # = [19, 19, 19, 19, 19, 19, 9, 9, 9],
        num_filters_list: List,  # = [16, 16, 16, 32, 48, 64, 64, 80, 80],
        add_avg_pool_list: List,  # = [1, 1, 1, 1, 1, 0, 0, 0, 0],
        sequential_part: Sequential = None,
    ) -> None:
        self.block_type = CvaeBlockEncoder
        self.in_shape = [sequence_length, num_features]
        if sequential_part is None:
            sequential_part = Sequential()
            sequential_part.add(InputLayer(input_shape=self.in_shape))
            # sequential_part.add(Masking(mask_value=0.)) #Mask zero values
        super().__init__(
            self.in_shape,
            filter_size_list,
            num_filters_list,
            add_avg_pool_list,
            sequential_part,
        )
        self.latent_dim = latent_dim
        if latent_dim != "None":
            self.sequential_part.add(Flatten())
            self.z_mean_layer = layers.Dense(latent_dim, name="z_mean")
            self.z_log_var_layer = layers.Dense(latent_dim, name="z_log_var")
            self.sampling = Sampling()

    def get_sequential_output_shape(self, sample):
        # out_shape_after_flatten = super().get_sequential_output_shape(sample)
        out_shape_before_flatten = self.sequential_part.layers[-2].output_shape
        return out_shape_before_flatten

    def latent_step(self, x: tf.Tensor) -> Union[tf.Tensor, dict]:
        """Perform latent step encoding and sampling

        Args:
            x (tf.Tensor): Input tensor

        Returns:
            tf.Tensor:
        """
        if self.latent_dim != "None":
            z_mean = self.z_mean_layer(x)
            z_log_var = self.z_log_var_layer(x)
            z = self.sampling([z_mean, z_log_var])
            return {"z_mean": z_mean, "z_log_var": z_log_var, "z": z}
        else:
            return x

    # def __call__(self, inputs, training=None, mask=None):
    #     encoded_vec = super().__call__(inputs, training, mask)
    #     return self.latent_step(encoded_vec)
    
    def call(self, inputs, training=None, mask=None):
        return super().call(inputs, training, mask)


if __name__ == "main":
    pass
