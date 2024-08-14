import tensorflow as tf
import numpy as np
import gc
import tqdm
from typing import List, Dict, Any

def masked_mse(data, reconstruction):
    # Return mse loss vector where the data is different from 0
    mask = data != 0
    mask = tf.cast(mask, dtype=data.dtype)
    squared_difference = tf.math.squared_difference(reconstruction, data)
    squared_difference *= mask
    max_axis = tf.size(tf.shape(data))
    x = tf.reduce_sum(squared_difference, axis=max_axis - 1)
    x = tf.math.divide_no_nan(x, tf.reduce_sum(mask, axis=max_axis - 1))
    reconstruction_loss = tf.reduce_sum(x, axis=1)
    return reconstruction_loss


def inverted_mask_contrast_mse(data, reconstruction, mask):
    # Ensure the mask is boolean
    mask = tf.cast(mask, tf.bool)
    mask_dense = tf.sparse.to_dense(mask)

    # Apply the mask to the data and the reconstruction
    data_dense = tf.sparse.to_dense(data)

    # Mask the ground-truth
    masked_data = tf.where(mask_dense, data_dense, tf.zeros_like(data_dense))
    
    # Invert the masked ground-truth
    masked_data = (
        1 * tf.cast(mask_dense, masked_data.dtype) - masked_data
    )
    
    # Mask the reconstruction
    masked_reconstruction = tf.where(
        mask_dense, reconstruction, tf.zeros_like(reconstruction)
    )


    # Compute MSE loss
    mse_loss = tf.reduce_sum(
        tf.square(masked_data - masked_reconstruction), axis=[1, 2, 3]
    ) / tf.reduce_sum(tf.cast(mask_dense, tf.float32), axis=[1, 2, 3])
    return mse_loss

def inverted_mask_contrast_mae(data, reconstruction, mask):
    # Ensure the mask is boolean
    mask = tf.cast(mask, tf.bool)
    
    # Convert sparse data to dense
    data_dense = tf.sparse.to_dense(data)
    
    # Convert sparse mask to dense
    mask_dense = tf.sparse.to_dense(mask)
    
    # Mask the ground-truth
    masked_data = tf.where(mask_dense, data_dense, tf.zeros_like(data_dense))
    
    # Invert the masked ground-truth
    masked_data = (
        1 * tf.cast(mask_dense, masked_data.dtype) - masked_data
    )
    # Mask the reconstruction
    masked_reconstruction = tf.where(mask_dense, reconstruction, tf.zeros_like(reconstruction))
    
    
    # Compute MAE loss
    mae_loss = tf.reduce_sum(
        tf.abs(masked_data - masked_reconstruction), axis=[1, 2, 3]
    ) / tf.reduce_sum(tf.cast(mask_dense, tf.float32), axis=[1, 2, 3])
    
    return mae_loss

def masked_mse_3d(data, reconstruction, mask):
    # Ensure the mask is boolean
    mask = tf.cast(mask, tf.bool)
    mask_dense = tf.sparse.to_dense(mask)
    # Apply the mask to the data and the reconstruction
    data_dense = tf.sparse.to_dense(data)
    reconstruction_dense = reconstruction
    masked_data = tf.where(mask_dense, data_dense, tf.zeros_like(data_dense))
    masked_reconstruction = tf.where(
        mask_dense, reconstruction_dense, tf.zeros_like(reconstruction_dense)
    )
    # Compute MSE loss
    mse_loss = tf.reduce_sum(
        tf.square(masked_data - masked_reconstruction), axis=[1, 2, 3]
    ) / tf.reduce_sum(tf.cast(mask_dense, tf.float32), axis=[1, 2, 3])
    return mse_loss


def compute_losses_for_dataset(model, dataset):
    # Initialize a dictionary to store the losses
    all_losses = {}
    # Iterate over each batch in the dataset
    for inputs in dataset:
        # Compute the loss
        loss_values = model.compute_loss(model(inputs, training=False))
        # Dynamically append the loss values for this batch to the corresponding lists
        for key, value in loss_values.items():
            if key not in all_losses:
                all_losses[key] = []
            all_losses[key].append(value.numpy())
    for k, v in all_losses.items():
        all_losses[k] = np.concatenate(v)
    return all_losses


def mse_timeseries(data, reconstruction):
    squared_difference = tf.math.squared_difference(
        data, reconstruction
    )  # SE between inputs and reconstructions
    max_axis = tf.size(tf.shape(data))
    x = tf.reduce_mean(
        squared_difference, axis=max_axis - 1
    )  # Average all the MSE between all the features/leads
    reconstruction_loss = tf.reduce_sum(x, axis=1)  # Average the values per timestep
    return reconstruction_loss


def weighted_mean_squared_error(x, y, w):
    squared_diff = tf.square(x - y)
    # mse = tf.linalg.matmul(squared_diff,tf.expand_dims(w,axis=-1))/tf.shape(squared_diff,out_type=float)[-1]
    weighted_squared_diff = squared_diff * w
    mse = tf.reduce_mean(weighted_squared_diff, axis=-1)
    return {
        "error": mse,
        "error_vector": squared_diff,
        "weighted_error_vector": weighted_squared_diff,
    }


def circular_squared_error(x, y):
    return tf.square(tf.cos(x) - tf.cos(y)) + tf.square(tf.sin(x) - tf.sin(y))


def squared_error(x, y):
    return tf.square(x - y)


def circular_weighted_mean_square_error(
    x: tf.Tensor, y: tf.Tensor, w: tf.Tensor, c: tf.Tensor
) -> tf.Tensor:
    """Compute MSE between x and y, where each pair in tensors x,y is weighted according to w.
    It also supports circular variables.

    Args:
        x (tf.Tensor): first input tensor
        y (tf.Tensor): second input tensor
        w (tf.Tensor): weight, same dimension of the last dimension of x,y
        c (tf.Tensor): float array where each element specify if the pair (x_i,y_i) is circular (c_i=1) or not (c_i=0)

    Returns:
        tf.Tensor: mse loss
    """
    se = squared_error(x, y)
    cse = circular_squared_error(x, y)
    e = c * cse + (1 - c) * se
    weighted_error = e * w
    mse = tf.reduce_mean(weighted_error, axis=-1)
    return {"error": mse, "error_vector": e, "weighted_error_vector": weighted_error}


def proximity_penalty(adjacency_matrix: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    """
    Computes the proximity penalty for a batch of weight vectors based on an adjacency matrix.

    Args:
        adjacency_matrix (tf.Tensor): A tensor of shape (n, n) representing the function of distances between sensors.
        weights (tf.Tensor): A tensor of shape (batch_size, n) representing the weights for each sensor in each batch.

    Returns:
        tf.Tensor: A tensor of shape (batch_size, 1) representing the proximity penalty for each batch.
    """
    # Compute the absolute differences between each pair of weights
    weight_diffs = tf.abs(tf.expand_dims(weights, 2) - tf.expand_dims(weights, 1))
    # Apply the adjacency matrix to the weight differences
    weighted_diffs = weight_diffs * adjacency_matrix
    # Sum the penalties for each batch
    penalty = tf.reduce_sum(weighted_diffs, axis=(1, 2))
    # Return the penalty as a vector of size batch_size x 1
    return tf.expand_dims(penalty, axis=1)

def compute_batch_avg_and_std(values: List[float]) -> Dict[str, float]:
    return {
        "avg": np.mean(values),
        "std": np.std(values)
    }

def merge_batch_stats(batch_stats: List[Dict[str, float]]) -> Dict[str, float]:
    all_avgs = [stat["avg"] for stat in batch_stats]
    all_stds = [stat["std"] for stat in batch_stats]
    combined_avg = np.mean(all_avgs)
    combined_std = np.sqrt(np.mean(np.square(all_stds)) + np.std(all_avgs)**2)
    return {
        "avg": combined_avg,
        "std": combined_std
    }

def compute_average_and_std(input_ds: Any, model: Any, batch_size: int, metrics: List[str]) -> Dict[str, Dict[str, float]]:
    """Computes average and std of the model metrics on the input dataset with batch size batch_size

    Args:
        input_ds (Any): The input dataset to evaluate.
        model (Any): The model used to compute the loss.
        batch_size (int): The size of batches to use from the dataset.
        metrics (List[str]): The list of metric names to compute.

    Returns:
        Dict[str, Dict[str, float]]: A dictionary where each metric name maps to another dictionary with keys 'avg' and 'std' for the average and standard deviation of that metric.
    """
    batch_stats = {metric: [] for metric in metrics}
    batched_ds = input_ds.batch(batch_size, drop_remainder=True)
    
    for sample in tqdm.tqdm(batched_ds):
        loss_dict = model.compute_loss(model(inputs=sample, training=False))
        for metric in metrics:
            batch_values = loss_dict[metric].numpy()
            batch_avg_std = compute_batch_avg_and_std(batch_values)
            batch_stats[metric].append(batch_avg_std)
        
        # Clear GPU memory
        del loss_dict
        tf.keras.backend.clear_session()
        gc.collect()
    
    result = {metric: merge_batch_stats(batch_stats[metric]) for metric in metrics}
    return result
if __name__ == "__main__":
    pass
