"""Module providing a fully connected tensorflow model
to perform ECGI
"""

import tensorflow as tf
from keras.layers import Dense, Reshape
from src.model.base import BaseModel
from src.model.nnmodels import get_dense_network


class DenseModel(BaseModel):
    """Fully connected model for ECG Imaging

    Args:
        BaseModel (_type_): Base class to perform ECGI
    """

    def __init__(
        self, units, sequence_length, num_features, output_size, dropout_rate, **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = len(units)
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.output_size = output_size
        # Model parameters
        self.sequential = get_dense_network(
            numNodesPerLayer=units,
            dropoutRate=dropout_rate,
            maskValue=0.0,
            addInFlatten=True,
        )
        self.sequential.add(Dense(tf.reduce_prod(output_size)))
        self.sequential.add(Reshape(output_size))

    def call(self, inputs, training):
        x = inputs["signal"]
        x = self.sequential(x)
        x = x[..., tf.newaxis]
        return {
            "signal": inputs["signal"],
            "acti_map": inputs["acti_map"],
            "conduct": inputs["conduct"],
            "mask": inputs["mask"],
            "predicted_acti_map": x,
        }


if __name__ == "__main__":
    pass
