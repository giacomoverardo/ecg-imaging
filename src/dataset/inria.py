import numpy as np
import os
import tensorflow as tf
import pyvista as pv
from typing import Dict, List, Tuple
from scipy.spatial.distance import cdist
from spektral.utils.convolution import gcn_filter
from src.utils.general import get_last_number_in_string
from sklearn.gaussian_process.kernels import RBF
import tqdm
from scipy.signal import find_peaks
def load_patient_ecgi_bacoyannis(patient_path: str | os.PathLike):
    patient_samples = []
    onsets = ["one_init_rv", "three_init", "two_init_lv"]
    # [os.path.join(dp, f) for dp, dn, filenames in os.walk(folder_path) for f in filenames]
    subfolders_real = [
        os.path.join(dp, d) for dp, dn, filenames in os.walk(patient_path) for d in dn
    ]
    onsets_real = [os.path.split(fol)[-1] for fol in subfolders_real]
    print(patient_path, set(onsets), set(onsets_real))
    assert set(onsets) == set(onsets_real)
    for i, (fol, onset) in enumerate(zip(subfolders_real, onsets_real)):
        files = [
            os.path.join(dp, f) for dp, dn, filenames in os.walk(fol) for f in filenames
        ]
        for f in files:
            sample_data_np = np.load(f, allow_pickle=True)
            sample_data_dict = {"onset": onset}
            for key in sample_data_np.files:
                sample_data_dict[key] = sample_data_np[key]
            patient_samples.append(sample_data_dict)
    return patient_samples


def load_ecgi_bacoyannis(datapath: str | os.PathLike = "./"):
    train_path = os.path.join(datapath, "simulation", "train")
    test_path = os.path.join(datapath, "simulation", "test")

    def get_split_dataset(path):
        sample_list = []
        patient_folder_list = [
            os.path.join(path, f.name) for f in os.scandir(path) if f.is_dir()
        ]
        for fol in patient_folder_list:
            patient_samples = load_patient_ecgi_bacoyannis(fol)
            patient_id = get_last_number_in_string(fol)
            for sample in patient_samples:
                new_sample = sample
                new_sample["patient_id"] = patient_id
                sample_list.append(new_sample)
        return sample_list

    train_ds = get_split_dataset(train_path)
    test_ds = get_split_dataset(test_path)
    return train_ds, test_ds

def get_peaks_location(signal, num_electrodes):
    extrema_ind = np.empty((num_electrodes,2), dtype=np.float64)
    for i in range(0,num_electrodes):
        electrode_signal = signal[i]
        peaks_ind = find_peaks(np.abs(electrode_signal),distance=250)[0]
        # peaks_val = electrode_signal[peaks_ind]
        extrema_ind[i,:] = np.sort(peaks_ind)
    mean_extrema_ind = np.mean(extrema_ind,axis=0)
    return mean_extrema_ind
# electrodes_mesh_dict = load_inria_electrodes_mesh("/mnt/giacomo/ecgi/data-bacoyannis/simulation")
def load_inria_npz(path, datapath, adj_matrix_threshold):
    # Convert path from tensor to string
    path = path.numpy().decode("utf-8")
    with np.load(path) as npz_data:
        # Extract numpy data from sample
        signal = npz_data["signal"].astype(np.float64)
        acti_map = npz_data["acti_map"].astype(np.float64)
        mask = npz_data["mask"].astype(np.int64)
        conduct = npz_data["conduct"].astype(np.float64)
        patient_id = path.split("/")[-3]
        onset = path.split("/")[-2]
        file_id = path.split("/")[-1].split(".")[0]
    # Get number of electrodes from signal
    num_electrodes = np.shape(signal)[0]
    # Get the estimate of peak locations (average between all electrodes signals)
    peaks_location = get_peaks_location(signal,num_electrodes)
    # Normalize the activation map between 0 and 1
    acti_map_min = np.min(acti_map)
    acti_map_max = np.max(acti_map)
    acti_map = (acti_map - acti_map_min) / (acti_map_max - acti_map_min)
    # Normalize the conduct map between 0 and 1
    conduct_min = np.min(conduct)
    conduct_max = np.max(conduct)
    conduct = (conduct - conduct_min) / (conduct_max - conduct_min)
    electrodes_mesh = load_inria_patient_electrodes_mesh(
        datapath.numpy().decode("utf-8"), patient_id
    ).astype(np.float64)
    # Compute adjaceny matrix and preprocess it with
    # adj_matrix = (
    #     cdist(electrodes_mesh, electrodes_mesh) < adj_matrix_threshold.numpy()
    # ).astype(np.float64)
    adj_matrix = RBF(length_scale=100,)(electrodes_mesh).astype(np.float64)
    np.fill_diagonal(adj_matrix, 0)
    # adj_matrix = gcn_filter(adj_matrix) # Add this when using GCN
    return (
        signal,
        acti_map,
        mask,
        conduct,
        electrodes_mesh,
        adj_matrix,
        peaks_location,
        patient_id,
        onset,
        file_id,
    )


def load_inria_npz_tensorflow(
    filepath: str | os.PathLike,
    datapath: str | os.PathLike,
    adj_matrix_threshold: float,
):
    return tf.py_function(
        load_inria_npz,
        [filepath, datapath, adj_matrix_threshold],
        [
            tf.float64,
            tf.float64,
            tf.int64,
            tf.float64,
            tf.float64,
            tf.float64,
            tf.string,
            tf.string,
            tf.string,
        ],
    )


def load_and_process_inria(
    file_path: str | os.PathLike,
    datapath: str | os.PathLike,
    adj_matrix_threshold: float,
) -> Dict:
    # Wrap NumPy logic in tf.py_function
    (
        signal,
        acti_map,
        mask,
        conduct,
        electrodes_mesh,
        adj_matrix,
        peaks_location,
        patient_id,
        onset,
        file_id,
    ) = tf.py_function(
        load_inria_npz,
        [file_path, datapath, adj_matrix_threshold],
        [
            tf.float64,
            tf.float64,
            tf.int64,
            tf.float64,
            tf.float64,
            tf.float64,
            tf.float64,
            tf.string,
            tf.string,
            tf.string,
        ],
    )
    acti_map = tf.sparse.from_dense(acti_map)
    mask = tf.sparse.from_dense(mask)
    conduct = tf.sparse.from_dense(conduct)
    # electrodes_mesh = tf.convert_to_tensor(electrodes_mesh, np.float64)
    # Set shapes if necessary (especially if using batching later)
    signal.set_shape((260, 450))
    # acti_map.set_shape((120, 120, 120))
    # mask.set_shape((120, 120, 120))
    # conduct.set_shape((120, 120, 120))
    acti_map = tf.sparse.reshape(acti_map, (120, 120, 120))
    mask = tf.sparse.reshape(mask, (120, 120, 120))
    conduct = tf.sparse.reshape(conduct, (120, 120, 120))
    electrodes_mesh.set_shape((260, 3))
    adj_matrix.set_shape((260, 260))
    peaks_location.set_shape((2,))
    return {
        "signal": signal,
        "acti_map": acti_map,
        "mask": mask,
        "conduct": conduct,
        "electrodes_mesh": electrodes_mesh,
        "adj_matrix": adj_matrix,
        "peaks_location":peaks_location,
        "patient_id": patient_id,
        "onset": onset,
        "file_id": file_id,
    }

@tf.autograph.experimental.do_not_convert
def get_inria_dataset_slice(
    basepath: str | os.PathLike,
    adj_matrix_threshold: float,
    slice: str,
    split: float = 0.0,
    seed: int = 0,
) -> Tuple[tf.data.Dataset, tf.data.Dataset] | tf.data.Dataset:
    path = os.path.join(basepath, slice, "*", "*", "*.npz")
    file_paths_dataset = tf.data.Dataset.list_files(path)
    if split > 0:
        num_files = file_paths_dataset.cardinality()
        split_size = int(num_files.numpy() * split)
        file_paths_dataset = file_paths_dataset.shuffle(buffer_size=1000, seed=seed)
        fp_ds1 = file_paths_dataset.skip(split_size)
        fp_ds2 = file_paths_dataset.take(split_size)
        ds1 = fp_ds1.map(
            lambda path: load_and_process_inria(path, basepath, adj_matrix_threshold),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds2 = fp_ds2.map(
            lambda path: load_and_process_inria(path, basepath, adj_matrix_threshold),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds1, ds2
    else:
        ds = file_paths_dataset.map(
            lambda path: load_and_process_inria(path, basepath, adj_matrix_threshold),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds


def get_inria_dataset(
    datapath: str | os.PathLike = "./",
    val_size: float = 0.0,
    seed: int = 0,
    adj_matrix_threshold: float = 50,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    train_ds, val_ds = get_inria_dataset_slice(
        basepath=datapath,
        slice="train",
        split=val_size,
        seed=seed,
        adj_matrix_threshold=adj_matrix_threshold,
    )
    test_ds = get_inria_dataset_slice(
        basepath=datapath, slice="test", adj_matrix_threshold=adj_matrix_threshold
    )
    return train_ds, val_ds, test_ds


def get_inria_sample_path(
    datapath: str | os.PathLike, slice: str, patient_id: str, onset: str, file_id: str
) -> str | os.PathLike:
    # Get sample path of an inria sample from information about the sample
    return os.path.join(datapath, slice, patient_id, onset, f"{file_id}.npz")


def get_inria_mesh_path(
    datapath: str | os.PathLike, patient_id: str, mesh: str
) -> str | os.PathLike:
    # Get mesh path of an inria sample from information about the sample
    return os.path.join(datapath, "mesh_sources", patient_id, mesh)


def get_inria_small_mesh_path(
    datapath: str | os.PathLike, patient_id: str
) -> str | os.PathLike:
    # Get small mesh path of an inria sample from information about the sample
    return get_inria_mesh_path(
        datapath=datapath, patient_id=patient_id, mesh="small_mesh.vtk"
    )


def get_inria_electrodes_mesh_path(
    datapath: str | os.PathLike, patient_id: str
) -> str | os.PathLike:
    # Get electrodes mesh path of an inria sample from information about the sample
    return get_inria_mesh_path(
        datapath=datapath, patient_id=patient_id, mesh="electrodes_mesh.vtk"
    )


def load_inria_electrodes_mesh(datapath: str | os.PathLike) -> Dict:
    mesh_path = os.path.join(datapath, "mesh_sources")
    electrodes_data = {}
    for root, dirs, files in os.walk(mesh_path):
        for file in files:
            if file == "electrodes_mesh.vtk":
                subfolder_name = os.path.basename(root)
                file_path = os.path.join(root, file)
                mesh = pv.read(file_path)
                points_array = mesh.points
                electrodes_data[subfolder_name] = points_array
    return electrodes_data


def load_inria_patient_electrodes_mesh(
    datapath: str | os.PathLike, patient_id
) -> np.ndarray:
    mesh_path = os.path.join(
        datapath, "mesh_sources", patient_id, "electrodes_mesh.vtk"
    )
    return np.array(pv.read(mesh_path).points)


def load_inria_signal_npz(path):
    with np.load(path) as npz_data:
        # Extract numpy data from sample
        signal = npz_data["signal"].astype(np.float64)
        patient_id = path.split("/")[-3]
        onset = path.split("/")[-2]
        file_id = path.split("/")[-1].split(".")[0]
    return signal, patient_id, onset, file_id


def load_inria_electrodes_signals(datapath: str | os.PathLike, slice: str) -> Dict:
    """Loads the electrodes signals from the inria dataset

    Args:
        datapath (str | os.PathLike): Data folder
        slice (str): subfolder (test or train)

    Returns:
        Dict: Dictionary containing a list of signals per patient
    """
    path = os.path.join(datapath, slice, "*", "*", "*.npz")
    file_paths = tf.io.gfile.glob(path)
    num_files = len(file_paths)
    temp_signal, _, _, _ = load_inria_signal_npz(file_paths[0])
    num_sensors, signal_len = np.shape(temp_signal)
    signals = np.zeros((num_files, num_sensors, signal_len), dtype=temp_signal.dtype)
    patient_ids = np.empty((num_files,), dtype=object)
    onsets = np.empty((num_files,), dtype=object)
    file_ids = np.empty((num_files,), dtype=object)
    for i, file_path in enumerate(file_paths):
        signal, patient_id, onset, file_id = load_inria_signal_npz(file_path)
        signals[i, :, :] = signal
        patient_ids[i] = patient_id
        onsets[i] = onset
        file_ids[i] = file_id
    return signals, patient_ids, onsets, file_ids

def get_inria_patient_folder(datapath: str | os.PathLike, patient_id: str)->os.PathLike:
    patient_to_slice_dict = {"Patient004": "train", 
                             "Patient005": "train", 
                             "Patient013": "train", 
                             "Patient014": "train",
                             "Patient015": "train",
                             "Patient011": "test",
                             "Patient012": "test"}
    slice = patient_to_slice_dict[patient_id]
    patient_folder = os.path.join(datapath, slice, patient_id)
    return patient_folder
def get_inria_patient_sample_paths(datapath: str | os.PathLike, patient_id: str) -> List[os.PathLike]:
    patient_folder = get_inria_patient_folder(datapath=datapath, patient_id=patient_id)
    sample_paths = []
    for root, _, files in os.walk(patient_folder):
        for file in files:
            if file.endswith('.npz'):
                sample_paths.append(os.path.join(root, file))
    return sample_paths

def get_geometry_info(sample_path: os.PathLike) -> Dict:
    pkg = {key: val for key, val in np.load(sample_path, allow_pickle=True).items()}
    return pkg["geometry_info"].item()

def check_geometry_info_consistency(sample_paths: List[os.PathLike]) -> bool:
    if not sample_paths:
        return True
    first_geometry_info = get_geometry_info(sample_paths[0])
    for sample_path in tqdm.tqdm(sample_paths[1:]):
        current_geometry_info = get_geometry_info(sample_path)
        if current_geometry_info != first_geometry_info:
            return False
    return True

def get_mask_from_path(sample_path: os.PathLike) -> np.ndarray:
    pkg = {key: val for key, val in np.load(sample_path, allow_pickle=True).items()}
    return pkg["mask"].astype(np.uint8)

def check_masks_consistency(sample_paths: List[os.PathLike]) -> bool:
    if not sample_paths:
        return True
    first_mask = get_mask_from_path(sample_paths[0])
    for sample_path in sample_paths[1:]:
        current_mask = get_mask_from_path(sample_path)
        if not np.array_equal(first_mask, current_mask):
            return False
    return True

def check_consistency_for_all_patients(datapath: str, patient_ids_list: List[str]) -> None:
    # Check consistency of geometries by running:
    # patient_ids_list = [f"Patient{num:03}" for num in [4, 5, 11, 12, 13, 14, 15]]
    # check_geometry_consistency_for_all_patients(cfg.datapath, patient_ids_list)
    for patient in patient_ids_list:
        patient_paths = get_inria_patient_sample_paths(datapath, patient)
        geometry_consistent = check_geometry_info_consistency(patient_paths)
        masks_consistent = check_masks_consistency(patient_paths)
        
        if not geometry_consistent:
            print(f"Geometry info is inconsistent for {patient}")
        if not masks_consistent:
            print(f"Masks are inconsistent for {patient}")
        if geometry_consistent and masks_consistent:
            print(f"Geometry info and masks are consistent for {patient}")

def get_geometry_info_per_patient_dict(datapath: str) -> Dict[str, dict]:
    patient_ids_list = [f"Patient{num:03}" for num in [4, 5, 11, 12, 13, 14, 15]]
    geometry_info_dict = {}
    
    for patient in patient_ids_list:
        patient_paths = get_inria_patient_sample_paths(datapath, patient)
        if patient_paths:
            geometry_info = get_geometry_info(patient_paths[0])
            geometry_info_dict[patient] = geometry_info
    
    return geometry_info_dict

def get_masks_per_patient_dict(datapath: str) -> Dict[str, np.ndarray]:
    patient_ids_list = [f"Patient{num:03}" for num in [4, 5, 11, 12, 13, 14, 15]]
    masks_dict = {}
    
    for patient in patient_ids_list:
        patient_paths = get_inria_patient_sample_paths(datapath, patient)
        if patient_paths:
            mask = get_mask_from_path(patient_paths[0])
            masks_dict[patient] = mask
    
    return masks_dict

if __name__ == "__main__":
    pass
