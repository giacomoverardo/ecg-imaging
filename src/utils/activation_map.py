import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import pyvista as pv

def compute_zero_value_clusters(activation_mesh: pv.core.pointset.UnstructuredGrid, q: float) -> tuple:
    """Compute clusters for points with activation values below a threshold using DBSCAN.
    
    Args:
        activation_mesh (pv.core.pointset.UnstructuredGrid): The mesh containing activation data.
        q (float): Quantile to determine the threshold for zero activation points.
        
    Returns:
        tuple: Zero value points, zero value activations, DBSCAN labels.
    """
    zero_acti_threshold = np.quantile(activation_mesh["activation"], q=q)
    zero_value_indexes = activation_mesh["activation"] <= zero_acti_threshold
    zero_value_points = activation_mesh.points[zero_value_indexes]
    zero_value_values = activation_mesh["activation"][zero_value_indexes]
    dbscan = DBSCAN(eps=3, min_samples=20).fit(zero_value_points)
    return zero_value_points, zero_value_values, dbscan.labels_

def extract_top_clusters(zero_value_points: np.ndarray, zero_value_values: np.ndarray, labels: np.ndarray, num_onsets: int) -> list:
    """Extract the top clusters based on the number of points in each cluster.
    
    Args:
        zero_value_points (np.ndarray): Points with activation values below the threshold.
        zero_value_values (np.ndarray): Activation values of the zero value points.
        labels (np.ndarray): Labels assigned by DBSCAN.
        num_onsets (int): Number of top clusters to extract.
        
    Returns:
        list: Surfaces of the top clusters.
    """
    label_values = np.unique(labels)
    num_labels_per_cluster = {label: np.sum(labels == label) for label in label_values}
    top_k_label_values = sorted(num_labels_per_cluster, key=num_labels_per_cluster.get, reverse=True)[:num_onsets]
    cluster_surfaces = []
    for label in top_k_label_values:
        if label != -1:
            cluster_indexes = labels == label
            cluster_points = zero_value_points[cluster_indexes]
            cluster_values = zero_value_values[cluster_indexes]
            cluster_surf = pv.wrap(cluster_points).delaunay_3d(alpha=0.001)
            cluster_surf["activation"] = cluster_values
            cluster_surfaces.append(cluster_surf)
    return cluster_surfaces

def compute_onsets_dbscan(activation_mesh:pv.core.pointset.UnstructuredGrid, num_onsets:int, q:float=0.1)->np.ndarray:
    """Compute onsets by means of DBSCAN clustering.

    Args:
        activation_mesh (pv.core.pointset.UnstructuredGrid): pyvista unstructured grid that contains the positions of the points of the map and their activation.
        num_onsets (int): Number of onsets to be detected
        q (float, optional): Quantile that represents the starting events. Values outside the quantile in activation_mesh are zeroed out. Defaults to 0.1.

    Returns:
        np.ndarray: Onsets 3D position. Each row represent one onset.
    """
    activation_map = activation_mesh["activation"] # Extract activation from mehs
    value_points = activation_mesh.points # Extract geometric 3D position from mesh

    zero_acti_threshold = np.quantile(activation_map, q=q)
    zero_value_indexes = activation_map <= zero_acti_threshold
    zero_value_points = value_points[zero_value_indexes]
    zero_value_values = activation_map[zero_value_indexes]
    dbscan = DBSCAN(eps=3, min_samples=20).fit(zero_value_points)
    # weights = 1.0-activation_map/np.max(activation_map) # Compute weights 
    # weights[weights<np.quantile(weights, q=1.0-q)]=0.0 # Keep only q% of the weights (the ones at the start)
    # dbscan = DBSCAN(eps=10, min_samples=20).fit(zero_value_points, sample_weight=weights)
    label_values = np.unique(dbscan.labels_)
    num_labels_per_cluster = {label: np.sum(dbscan.labels_==label) for label in label_values}
    top_k_label_values = sorted(num_labels_per_cluster, key=num_labels_per_cluster.get, reverse=True)[:num_onsets]
    onsets = []
    for label in top_k_label_values:
        if label != -1:
            cluster_indexes = dbscan.labels_ == label
            cluster_points = zero_value_points[cluster_indexes]
            cluster_values = zero_value_values[cluster_indexes]
            argmin_cluster_value = np.argmin(cluster_values)
            cluster_onset = cluster_points[argmin_cluster_value,:]
            onsets.append(cluster_onset)
    onsets = np.stack(onsets,axis=0)
    return onsets

def compute_onsets_kmeans(activation_mesh:pv.core.pointset.UnstructuredGrid, num_onsets:int, q:float=0.1)->np.ndarray:
    """Compute onsets positions by means of K-Means clustering.

    Args:
        activation_mesh (pv.core.pointset.UnstructuredGrid): pyvista unstructured grid that contains the positions of the points of the map and their activation.
        num_onsets (int): Number of onsets to be detected
        q (float, optional): Quantile that represents the starting events. Values outside the quantile in activation_mesh are zeroed out. Defaults to 0.1.

    Returns:
        np.ndarray: Onsets 3D position. Each row represents one onset.
        np.ndarray: 3D positions of cluster centers. Each row represents one cluster center.
    """
    activation_map = activation_mesh["activation"] # Extract activation from mehs
    zero_value_points = activation_mesh.points # Extract geometric 3D position from mesh
    weights = 1.0-activation_map/np.max(activation_map) # Compute weights 
    weights[weights<np.quantile(weights, q=1.0-q)]=0.0 # Keep only q% of the weights (the ones at the start)
    kmeans = KMeans(n_clusters=num_onsets, random_state=0, n_init="auto").fit(zero_value_points, sample_weight=weights)
    cluster_centers = kmeans.cluster_centers_
    # Compute closest point (Euclidean distance) to cluster centers and use it as onset position
    onsets = []
    for cluster_c in cluster_centers:
        closest_point_to_cluster_index = np.argmin(np.sum(np.square(zero_value_points-cluster_c),axis=1))
        onsets.append(zero_value_points[closest_point_to_cluster_index])
    onsets = np.stack(onsets,axis=0)
    return onsets, cluster_centers

if __name__ == '__main__':
    pass