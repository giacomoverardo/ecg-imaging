import os
import vtk
import numpy as np
import SimpleITK as sitk
import pyvista as pv
from tetgen import TetGen
import copy
from typing import List, Dict
from src.utils.activation_map import compute_zero_value_clusters, extract_top_clusters, compute_onsets_kmeans
from src.dataset.inria import (
    get_inria_sample_path,
    get_inria_small_mesh_path,
    get_geometry_info_per_patient_dict,
    get_masks_per_patient_dict
)

# Parts of this code has been made available by Buntheng Ly https://www-sop.inria.fr/asclepios/biblio/Author/LY-B.html


def get_neighbor(index):
    """ """
    x, y, z = index
    return [
        [x - 1, y - 1, z - 1],
        [x, y - 1, z - 1],
        [x + 1, y - 1, z - 1],
        [x - 1, y, z - 1],
        [x, y, z - 1],
        [x + 1, y, z - 1],
        [x - 1, y + 1, z - 1],
        [x, y + 1, z - 1],
        [x + 1, y + 1, z - 1],
        [x - 1, y - 1, z],
        [x, y - 1, z],
        [x + 1, y - 1, z],
        [x - 1, y, z],
        [x + 1, y, z],
        [x - 1, y + 1, z],
        [x, y + 1, z],
        [x + 1, y + 1, z],
        [x - 1, y - 1, z + 1],
        [x, y - 1, z + 1],
        [x + 1, y - 1, z + 1],
        [x - 1, y, z + 1],
        [x, y, z + 1],
        [x + 1, y, z + 1],
        [x - 1, y + 1, z + 1],
        [x, y + 1, z + 1],
        [x + 1, y + 1, z + 1],
    ]


def get_euclidean_dist(point_a, point_b):
    """ """
    return np.sum((np.array(point_a) - np.array(point_b)) ** 2) ** 0.5


def find_closest(pos, mask_img):
    """ """
    # get index
    index = mask_img.TransformPhysicalPointToIndex(pos)

    if mask_img[index]:
        return index

    def get_val(index, img, c=0):
        try:
            return img[index]
        except:
            return c

    neighbor_index = get_neighbor(index)
    neighbor_val = [get_val(_index, mask_img) for _index in neighbor_index]

    valid_index = [index for index, val in zip(neighbor_index, neighbor_val) if val]
    if len(valid_index) == 0:
        return None

    elif len(valid_index) == 1:
        return valid_index[0]

    else:
        cont_index = mask_img.TransformPhysicalPointToContinuousIndex(pos)
        neighbor_dist = [
            get_euclidean_dist(cont_index, _index) for _index in valid_index
        ]
        return valid_index[np.argmin(neighbor_dist)]


def get_link(mask_img, mesh):
    """Get correspoding voxel indices to points on mesh."""
    n_points = mesh.GetNumberOfPoints()
    link = np.zeros((n_points, 3)) * np.nan
    missing = []
    locator = vtk.vtkPointLocator()
    points = vtk.vtkPoints()
    locator.InitPointInsertion(points, mesh.GetBounds())
    locator_index = []

    for point_id in range(n_points):
        pos = mesh.GetPoint(point_id)
        index = find_closest(pos=pos, mask_img=mask_img)

        if index is not None:
            link[point_id] = index
            locator.InsertNextPoint(pos)
            locator_index.append(index)

        else:
            missing.append([pos, point_id])

    for pos, point_id in missing:
        closest_id = locator.FindClosestInsertedPoint(pos)
        index = locator_index[closest_id]

        locator.InsertNextPoint(pos)
        locator_index.append(index)

        link[point_id] = index
    return link.astype(int).tolist()


def to_tetrahedral(polydata):
    # Convert surface mesh to tetrahedral mesh ()
    """Convert surface polydata to tetrahedral mesh (filled)."""
    pvsurf = pv.wrap(polydata)
    tet = TetGen(pvsurf)
    tet.make_manifold()
    tet.tetrahedralize()
    polydata = tet.grid  # type(res) == vtkUnstructureGrid
    return polydata


def save_acti_map_fig(
    acti_map: np.ndarray,
    data_path: str,
    save_path: str,
    slice: str,
    patient_id: str,
    onset: str,
    file_id: str,
    add_text,
):
    # Parts of this code has been made available by Buntheng Ly https://www-sop.inria.fr/asclepios/biblio/Author/LY-B.html
    # patient_id, onset, file_id = patient_ids[i].numpy().decode('utf-8'), onsets[i].numpy().decode('utf-8'), file_ids[i].numpy().decode('utf-8')

    sample_path = get_inria_sample_path(
        datapath=data_path,
        slice=slice,
        patient_id=patient_id,
        onset=onset,
        file_id=file_id,
    )
    pkg = {key: val for key, val in np.load(sample_path, allow_pickle=True).items()}
    pkg["geometry_info"] = pkg["geometry_info"].item()
    acti_map_min = np.min(pkg["acti_map"])
    acti_map_max = np.max(pkg["acti_map"])
    # Convert ndarray to sitk.Image
    mask = pkg["mask"].astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask)
    # set up physical information
    mask_img.SetOrigin(pkg["geometry_info"]["resize"]["Origin"])
    mask_img.SetSpacing(pkg["geometry_info"]["resize"]["Spacing"])
    mask_img.SetDirection(pkg["geometry_info"]["resize"]["Direction"])
    acti = acti_map.astype(np.float64) * (acti_map_max - acti_map_min) + acti_map_min
    acti[~mask.astype(bool)] = np.nan
    acti_img = sitk.GetImageFromArray(acti)
    # set up physical information
    acti_img.SetOrigin(pkg["geometry_info"]["resize"]["Origin"])
    acti_img.SetSpacing(pkg["geometry_info"]["resize"]["Spacing"])
    acti_img.SetDirection(pkg["geometry_info"]["resize"]["Direction"])

    # Create an referece image in original geometry
    ref_img = sitk.Image(pkg["geometry_info"]["orig"]["Size"], sitk.sitkUInt8)
    ref_img.SetOrigin(pkg["geometry_info"]["orig"]["Origin"])
    ref_img.SetSpacing(pkg["geometry_info"]["orig"]["Spacing"])
    ref_img.SetDirection(pkg["geometry_info"]["orig"]["Direction"])

    transformer = sitk.Transform()
    transformer.SetIdentity()
    revert_mask_img = sitk.Resample(
        mask_img, ref_img, transformer, sitk.sitkNearestNeighbor
    )

    revert_acti_img = sitk.Resample(
        acti_img, ref_img, transformer, sitk.sitkNearestNeighbor
    )

    surf_polydata = pv.read(
        get_inria_small_mesh_path(datapath=data_path, patient_id=patient_id)
    )
    tetra_polydata = to_tetrahedral(surf_polydata)

    surf_links = get_link(revert_mask_img, surf_polydata)
    tetra_links = get_link(revert_mask_img, tetra_polydata)
    surf_polydata["activation"] = [revert_acti_img[l] for l in surf_links]
    tetra_polydata["activation"] = [revert_acti_img[l] for l in tetra_links]

    file_name = f"acti_map_{patient_id}_{onset}_{file_id}_{add_text}"
    # surf_polydata.save(os.path.join(save_path,f"{file_name}_surface.vtk"))
    tetra_polydata.save(os.path.join(save_path, f"{file_name}_tetra.vtk"))

def find_min_max_activation_point(acti_map_list:List[np.ndarray], 
                              acti_mask:np.ndarray, 
                              tetra_links, 
                              tetra_polydata,
                              ref_img,
                              geometry_info, transformer,
                              acti_map_min:float=None, acti_map_max:float=None):
    # Define a function to compute the minimum activation point of the map
    # Optionally, it can also compute the temporal value (the problem is that in that case the map should be un-normalized)
    results_list = []
    for acti_map in acti_map_list:
        # set up physical information
        acti = acti_map.astype(np.float64)
        acti[~acti_mask.astype(bool)] = np.nan
        acti_img = sitk.GetImageFromArray(acti)
        acti_img.SetOrigin(geometry_info["resize"]["Origin"])
        acti_img.SetSpacing(geometry_info["resize"]["Spacing"])
        acti_img.SetDirection(geometry_info["resize"]["Direction"])
        revert_acti_img = sitk.Resample(
            acti_img, ref_img, transformer, sitk.sitkNearestNeighbor
        )
        # Get minimum activation point
        tetra_act = np.array([revert_acti_img[mylink] for mylink in tetra_links])
        min_surf_act = np.min(tetra_act[tetra_act>0])
        argmin_surf_act = np.where(tetra_act==min_surf_act)
        argmin_surf_pos = tetra_polydata.points[argmin_surf_act]
        # Get maximum activation point
        max_surf_act = np.max(tetra_act[tetra_act>0])
        argmax_surf_act = np.where(tetra_act==max_surf_act)
        argmax_surf_pos = tetra_polydata.points[argmax_surf_act]
        results_list.append({"argmin_surf_pos":argmin_surf_pos, "argmax_surf_pos":argmax_surf_pos, 
            "argmin_surf_act": argmin_surf_act, "argmax_surf_act":argmax_surf_act,})
    return results_list

def find_min_activation_point_k_means(acti_map:np.ndarray, 
                              num_onsets:int,     
                              acti_mask:np.ndarray, 
                              tetra_links, 
                              tetra_polydata,
                              ref_img,
                              geometry_info, transformer):
    # Define a function to compute the minimum activation point of the map
    # Optionally, it can also compute the temporal value (the problem is that in that case the map should be un-normalized)
    # set up physical information
    acti = acti_map.astype(np.float64)
    acti[~acti_mask.astype(bool)] = np.nan
    acti_img = sitk.GetImageFromArray(acti)
    acti_img.SetOrigin(geometry_info["resize"]["Origin"])
    acti_img.SetSpacing(geometry_info["resize"]["Spacing"])
    acti_img.SetDirection(geometry_info["resize"]["Direction"])
    revert_acti_img = sitk.Resample(
        acti_img, ref_img, transformer, sitk.sitkNearestNeighbor
    )
    # Get minimum activation point
    surf_act = np.array([revert_acti_img[mylink] for mylink in tetra_links])
    surf_polydata_copy = copy.deepcopy(tetra_polydata)
    surf_polydata_copy["activation"] = surf_act
    onsets_positions, cluster_centers = compute_onsets_kmeans(activation_mesh=surf_polydata_copy, num_onsets=num_onsets, q=0.1)
    return onsets_positions, cluster_centers

def create_tetra_links_and_reference_images(datapath: str) -> Dict[str, any]:
    patient_ids_list = [f"Patient{num:03}" for num in [4, 11, 12, 13, 14, 15]]
    
    # Obtain geometry info and masks
    geometry_info_dict = get_geometry_info_per_patient_dict(datapath)
    masks_dict = get_masks_per_patient_dict(datapath)
    
    # Obtain meshes
    mesh_dict = {
        patient_id: pv.read(get_inria_small_mesh_path(datapath, patient_id))
        for patient_id in patient_ids_list
    }
    
    tetra_links_dict = {}
    reference_images_dict = {}
    
    for patient in patient_ids_list:
        geometry_info = geometry_info_dict[patient]
        mask = masks_dict[patient]
        
        # Convert ndarray to sitk.Image
        mask_img = sitk.GetImageFromArray(mask)
        mask_img.SetOrigin(geometry_info["resize"]["Origin"])
        mask_img.SetSpacing(geometry_info["resize"]["Spacing"])
        mask_img.SetDirection(geometry_info["resize"]["Direction"])
        
        # Create a reference image in original geometry
        ref_img = sitk.Image(geometry_info["orig"]["Size"], sitk.sitkUInt8)
        ref_img.SetOrigin(geometry_info["orig"]["Origin"])
        ref_img.SetSpacing(geometry_info["orig"]["Spacing"])
        ref_img.SetDirection(geometry_info["orig"]["Direction"])
        
        transformer = sitk.Transform()
        transformer.SetIdentity()
        
        revert_mask_img = sitk.Resample(
            mask_img, ref_img, transformer, sitk.sitkNearestNeighbor
        )
        
        surf_polydata = mesh_dict[patient]
        tetra_polydata = to_tetrahedral(surf_polydata)
        # surf_links = get_link(revert_mask_img, surf_polydata)
        tetra_links = get_link(revert_mask_img, tetra_polydata)
        
        tetra_links_dict[patient] = tetra_links
        reference_images_dict[patient] = ref_img
    
    return tetra_links_dict, reference_images_dict

def plot_bdscan_surfaces(activation_mesh: pv.core.pointset.UnstructuredGrid, onsets_real: np.ndarray, num_onsets: int, q: float = 0.1) -> None:
    """Computes clusters with DBSCAN and plots them as surfaces.
    
    Args:
        activation_mesh (pv.core.pointset.UnstructuredGrid): Mesh containing activation data.
        onsets_real (np.ndarray): Array of real onset points.
        num_onsets (int): Number of top clusters to plot.
        q (float, optional): Quantile to determine the threshold for zero activation points. Defaults to 0.1.
    """
    zero_value_points, zero_value_values, labels = compute_zero_value_clusters(activation_mesh, q)
    cluster_surfaces = extract_top_clusters(zero_value_points, zero_value_values, labels, num_onsets)
    
    p = pv.Plotter()
    p.add_mesh(activation_mesh, opacity=0.1, color='white')
    for surf in cluster_surfaces:
        p.add_mesh(surf, scalars="activation", show_edges=False)
    p.add_mesh(onsets_real, color="#FF0000", render_points_as_spheres=True, point_size=10)
    for center in onsets_real:
        p.add_mesh(pv.Sphere(radius=10, center=center), color="red", opacity=0.5)
    p.view_xz()
    p.show()

if __name__ == "__main__":
    pass
