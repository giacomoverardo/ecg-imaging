
import tensorflow as tf
import keras
import numpy as np
from keras import backend as K
from typing import Tuple, List
class Squeeze(keras.layers.Layer):
    def __init__(self):
        super(Squeeze, self).__init__()

    def call(self, inputs):
        return tf.squeeze(inputs)

def get_truncated_model(model:tf.keras.Model, num_layers):
    truncated_model = tf.keras.models.clone_model(model)
    with tf.device('cpu'):
        model_inputs = keras.Input(shape=model.layers[0].input_shape[0][1:])
        x = model_inputs
        for i in range(num_layers):
            truncated_model.layers[i].set_weights(model.layers[i].get_weights()) 
            x = truncated_model.layers[i](x)
        new_model = tf.keras.Model(inputs = model_inputs, outputs = x)
    return new_model

def bounded_output(x, lower, upper):
    # https://stackoverflow.com/questions/62562463/constraining-a-neural-networks-output-to-be-within-an-arbitrary-range
    scale = upper - lower
    return scale * tf.nn.sigmoid(x) + lower

def get_torch_nn_parameters(model, p_type:str="all")->int:
    """Return number of all/trainable/non-trainable parameters in input model

    Args:
        model (Module): input torch model
        p_type (str, optional): choose between all/trainable/non-trainable parameters. Defaults to "all".

    Raises:
        ValueError: when p_type is not all/trainable/non-trainable
        
    Returns:
        int: number of all/trainable/non-trainable parameters 
    """
    if(p_type=="all"):
        p_filter = lambda p: True 
    elif(p_type=="trainable"):
        p_filter = lambda p: p.requires_grad
    elif(p_type=="non-trainable"):
        p_filter = lambda p: not(p.requires_grad)
    else:
        raise ValueError("Argument p filter shold be all/trainable/non-trainable")
    model_parameters = filter(p_filter,model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    return num_params
    
def get_parameters_count_from_model(in_model):
    trainable_count = int(np.sum([K.count_params(w) for w in in_model.trainable_weights]))
    non_trainable_count = int(np.sum([K.count_params(w) for w in in_model.non_trainable_weights]))
    total_num_params = trainable_count + non_trainable_count
    parameters_count_dict = {
        "num_trainable": trainable_count,
        "num_non_trainable": non_trainable_count,
        "num_parameters": total_num_params
    }
    return parameters_count_dict

def top_k_and_bottom_k_indices(tensor: tf.Tensor, k: int) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Get the indices of the top k highest values and the bottom k lowest values for each row in a tensor.

    Args:
        tensor (tf.Tensor): A 2D tensor with shape (batch_size, num_features).
        k (int): The number of top and bottom indices to retrieve for each row.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing two tensors:
            - top_k_indices: Indices of the top k highest values for each row.
            - bottom_k_indices: Indices of the bottom k lowest values for each row.
    """
    # Get the top k indices for each row
    top_k_values, top_k_indices = tf.nn.top_k(tensor, k=k, sorted=True)
    
    # Get the bottom k indices for each row
    bottom_k_values, bottom_k_indices = tf.nn.top_k(-tensor, k=k, sorted=True)
    
    return top_k_indices, bottom_k_indices

def zero_out_indices(tensor: tf.Tensor, indices: tf.Tensor) -> tf.Tensor:
    """
    Set the values of the tensor to zero at the given indices for each row.

    Args:
        tensor (tf.Tensor): A 2D or 3D tensor with shape (batch_size, num_features) or (batch_size, num_features, num_timestamps).
        indices (tf.Tensor): A 2D tensor with shape (batch_size, k), containing the indices to set to zero for each row.

    Returns:
        tf.Tensor: The modified tensor with the specified indices set to zero.
    """
    tensor_rank = tf.rank(tensor)
    # assert tensor_rank==2 or tensor_rank==3
    batch_size = tf.shape(tensor)[0]
    k = tf.shape(indices)[1]
    if(tensor_rank==3):
        num_timestamps = tf.shape(tensor)[-1]
        scatter_updates = tf.zeros([batch_size * k, num_timestamps])
    elif(tensor_rank==2):
        scatter_updates = tf.zeros([batch_size * k])
    else:
        scatter_updates = tf.zeros([batch_size * k])
    mask = tf.ones_like(tensor)
    # Create indices for scattering
    scatter_indices = tf.stack([tf.expand_dims(tf.range(batch_size), axis=1) * tf.ones([1, k], dtype=tf.int32), indices], axis=2)
    scatter_indices = tf.reshape(scatter_indices, [-1, 2])

    # Scatter zeros into the mask at the specified indices
    mask = tf.tensor_scatter_nd_update(
        mask, 
        tf.reshape(scatter_indices, [-1, 2]), 
        scatter_updates
    )

    # Apply the mask to the tensor
    tensor = tensor * mask
    
    return tensor

if __name__ == '__main__':
    pass