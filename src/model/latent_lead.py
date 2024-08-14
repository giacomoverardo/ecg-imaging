"""Module providing a tensorflow model for ECG imaging based on
a GNN and FMM-based encoder
"""

from typing import List, Tuple, Union
import tensorflow as tf
import numpy as np
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
from keras.metrics import Mean
from src.model.vae import Sampling
from src.model.baseFMMAE import FMM_head
from src.model.base import BaseModel
from src.model.gnn import GCNModel
from src.model.decoder3d import Decoder3D
from src.utils.metrics import inverted_mask_contrast_mse, masked_mse, inverted_mask_contrast_mae
from src.utils.fmm import get_all_lead_indexes_circular, get_lead_indexes_circular
from src.utils.nn import top_k_and_bottom_k_indices, zero_out_indices

class LatentLead(BaseModel):
    """Model for ECG imaging based on a GNN and FMM-based encoder

    Args:
        BaseModel (_type_): Base model class to perform ECGI
    """

    def __init__(
        self,
        num_features,
        num_waves,
        sequence_length,
        max_omega,
        batch_size,
        reconstruction_acti_weight,
        reconstruction_signal_weight,
        alpha_loss_weight,
        encoder,
        decoder,
        gnn,
        *args,
        **kwargs
    ):
        super().__init__(*args,**kwargs)
        # Train trackers
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_acti_loss_tracker = Mean(name="rec_acti_loss")
        self.reconstruction_signal_loss_tracker = Mean(name="rec_signal_loss")
        self.reconstruction_mae_acti_loss_tracker = Mean(name="mae_rec_acti_loss")
        self.alpha_loss_tracker = Mean(name="alpha_loss")
        # Test trackers
        self.test_total_loss_tracker = Mean(name="test_total_loss")
        self.test_reconstruction_acti_loss_tracker = Mean(name="test_rec_acti_loss")
        self.test_reconstruction_signal_loss_tracker = Mean(name="test_rec_signal_loss")
        self.test_reconstruction_mae_acti_loss_tracker = Mean(name="test_mae_rec_acti_loss")
        self.test_alpha_loss_tracker = Mean(name="test_alpha_loss")

        self.global_avg_pooling = Flatten()  # tf.keras.layers.GlobalAveragePooling1D()
        self.num_leads = num_features
        self._num_kept_leads = num_features
        self.num_waves = num_waves
        self.seq_len = sequence_length
        self.max_omega = max_omega
        self.batch_size = batch_size
        self.reconstruction_acti_weight = reconstruction_acti_weight
        self.reconstruction_signal_weight = reconstruction_signal_weight
        self.alpha_loss_weight = alpha_loss_weight
        # Create encoder to encode input ecg data to latent space
        self.encoder = encoder
        self.fmm_head = FMM_head(
            num_leads=num_features,
            num_waves=num_waves,
            seq_len=sequence_length,
            max_omega=max_omega,
            batch_size=batch_size,
        )
        # Compute number of not_shared fmm parameters per lead
        self.num_not_shared_parameters_per_lead = len(
            get_lead_indexes_circular(0, num_features, num_waves, include_shared=False)
        )
        # Compute the indexes of the fmm parameters in the latent space for each lead
        # This function also concatenates the rows in a single row
        # Each row would have "self.num_not_shared_parameters_per_lead" elements
        self.lead_indexes_tensor = get_all_lead_indexes_circular(
            num_leads=num_features, num_waves=num_waves
        )
        # Add a dimension to be able to extract the correspondent FMM parameters from latent space during call
        self.lead_indexes_tensor = tf.expand_dims(self.lead_indexes_tensor, -1)
        # Create decoder to reconstruct activation map
        self.decoder = decoder
        # Create a GNN to compute the lead weights to be applied to the FMM parameters in the latent space
        self.gnn = gnn

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_acti_loss_tracker,
            self.reconstruction_signal_loss_tracker,
            self.reconstruction_mae_acti_loss_tracker,
            self.alpha_loss_tracker,
            self.test_total_loss_tracker,
            self.test_reconstruction_acti_loss_tracker,
            self.test_reconstruction_signal_loss_tracker,
            self.test_reconstruction_mae_acti_loss_tracker,
            self.test_alpha_loss_tracker,
        ]
    
    @property
    def num_kept_leads(self):
        return self._num_kept_leads

    @num_kept_leads.setter
    def num_kept_leads(self, value):
        if isinstance(value, int):
            self._num_kept_leads = value
        else:
            raise ValueError("num_kept_leads must be an integer")
        
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
            loss_dict = self.compute_loss(self(inputs=data, training=True))
            total_loss_mean = tf.reduce_mean(loss_dict["total_loss"])
        grads = tape.gradient(total_loss_mean, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(loss_dict["total_loss"])
        self.reconstruction_acti_loss_tracker.update_state(loss_dict["acti_loss"])
        self.reconstruction_signal_loss_tracker.update_state(loss_dict["signal_loss"])
        self.reconstruction_mae_acti_loss_tracker.update_state(loss_dict["mae_acti_loss"])
        self.alpha_loss_tracker.update_state(loss_dict["alpha_loss"])
        return {
            "loss": self.total_loss_tracker.result(),
            "acti_loss": self.reconstruction_acti_loss_tracker.result(),
            "signal_loss": self.reconstruction_signal_loss_tracker.result(),
            "mae_acti_loss": self.reconstruction_mae_acti_loss_tracker.result(),
            "alpha_loss":self.alpha_loss_tracker.result(),
        }

    def test_step(self, data):
        loss_dict = self.compute_loss(self(inputs=data, training=False))
        self.test_total_loss_tracker.update_state(loss_dict["total_loss"])
        self.test_reconstruction_acti_loss_tracker.update_state(loss_dict["acti_loss"])
        self.test_reconstruction_signal_loss_tracker.update_state(loss_dict["signal_loss"])
        self.test_reconstruction_mae_acti_loss_tracker.update_state(loss_dict["mae_acti_loss"])
        self.test_alpha_loss_tracker.update_state(loss_dict["alpha_loss"])
        return {
            "loss": self.test_total_loss_tracker.result(),
            "acti_loss": self.test_reconstruction_acti_loss_tracker.result(),
            "signal_loss": self.test_reconstruction_signal_loss_tracker.result(),
            "mae_acti_loss": self.test_reconstruction_mae_acti_loss_tracker.result(),
            "alpha_loss":self.test_alpha_loss_tracker.result(),
        }

    def compute_loss(self, inputs):
        # Compute reconstruction loss for the activation_map
        acti_map = inputs["acti_map"]
        predicted_acti_map = inputs["predicted_acti_map"]
        mask = inputs["mask"]
        acti_loss = inverted_mask_contrast_mse(acti_map, predicted_acti_map, mask)
        acti_loss_mae = inverted_mask_contrast_mae(acti_map, predicted_acti_map, mask)
        # Compute reconstruction loss for BSPM singals reconstructed with FMM_Head
        predicted_signal = inputs["predicted_signal"]
        signal = inputs["signal"]
        signal_loss = masked_mse(signal, predicted_signal)
        # Compute peaks localization loss
        peaks_location = (inputs["peaks_location"]/self.seq_len)*2*np.pi
        alpha_values = self.fmm_head.get_all_alpha_linear(inputs["predicted_fmm_coefficients_scaled"])
        alpha_loss = tf.reduce_sum(tf.square(peaks_location-alpha_values),axis=-1)
        # Compute total loss
        total_loss = (
            self.reconstruction_signal_weight * signal_loss
            + self.reconstruction_acti_weight * acti_loss
            + self.alpha_loss_weight * alpha_loss
        )
        return {"total_loss": total_loss, "signal_loss": signal_loss, 
                "acti_loss":acti_loss, "mae_acti_loss":acti_loss_mae, 
                "alpha_loss":alpha_loss}

    def call(self, inputs, training):
        x = inputs["signal"]  # [num_features, sequence_length]
        # Apply GNN to feature inputs and adjacency matrix to compute lead_weights
        lead_weights = self.gnn(inputs)
        if(self.num_kept_leads<self.num_leads):
            # Zero out the lowest weights 
            num_remaining_leads = self.num_leads-self.num_kept_leads
            _, bottom_k_indices = tf.nn.top_k(-lead_weights, k=num_remaining_leads, sorted=True)
            lead_weights_zeroed = zero_out_indices(lead_weights, bottom_k_indices)
            # Zero out the input leads in signal which correspond to the lowest weights 
            x = zero_out_indices(x, bottom_k_indices)
        else:
            lead_weights_zeroed = lead_weights
        
        # Swap signal feature and time axis because the encoder input has been
        # set in this order at initialization: [sequence_length, num_features]
        x = tf.experimental.numpy.swapaxes(x, 1, 2)
        # Encode parameters to latent space (before FMM-Head)
        x = self.encoder(inputs=x, training=training)
        if self.global_avg_pooling is not None:
            x = self.global_avg_pooling(x)
        # Apply FMM-Head to the latent space
        fmm_head_output_dict = self.fmm_head(
            x,
            x_len=self.seq_len,
            return_parameters_array=True,
            return_parameters_array_bounded=True,
        )
        # Extract bounded FMM parameters from latent space (still containing sine and cosine for circular parameters)
        parameters_array_bounded = fmm_head_output_dict["parameters_array_bounded"]
        # Repeat lead_weights to suit size of parameters_array_bounded
        lead_weights_repeat = tf.repeat(
            lead_weights_zeroed, repeats=self.num_not_shared_parameters_per_lead, axis=-1
        )
        # Create a mask of ones (we do not modify the parameters not present in self.lead_indexes_tensor)
        mask = tf.ones_like(parameters_array_bounded)
        # Modify mask to add the weight of each parameter in the correct index in the mask
        # Basically, the columns from lead_weights_repeat are moved to the columns of mask
        # according to self.lead_indexes_tensor
        mask = tf.transpose(
            tf.tensor_scatter_nd_update(
                tf.transpose(mask), # Write inside the mask sensor
                self.lead_indexes_tensor,   # Override only the parameters not shared among leads (leave 1 otherwise)
                tf.transpose(lead_weights_repeat),
            )
        )
        weighted_parameters_array_bounded = tf.math.multiply(
            mask, parameters_array_bounded
        )

        predicted_signal = fmm_head_output_dict["output"]
        predicted_activation_map = self.decoder(
            weighted_parameters_array_bounded,
            inputs["conduct"],
            inputs["signal"],
            training=training,
        )
        return {
            "acti_map": inputs["acti_map"],
            "predicted_acti_map": predicted_activation_map,
            "mask": inputs["mask"],
            "signal": tf.experimental.numpy.swapaxes(inputs["signal"], 1, 2),
            "predicted_signal": predicted_signal,
            "predicted_fmm_coefficients": fmm_head_output_dict["parameters_array"],
            "predicted_fmm_coefficients_scaled": fmm_head_output_dict[
                "parameters_array_bounded"
            ],
            "peaks_location":inputs["peaks_location"],
            "lead_weights": lead_weights,
            "patient_id": inputs["patient_id"],
        }


if __name__ == "__main__":
    pass
