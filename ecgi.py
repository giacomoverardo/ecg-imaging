# %%
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose
from hydra.utils import instantiate, call
import logging 
import pickle
import random
import time
import json
import tqdm
import SimpleITK as sitk
import pyvista as pv
from src.utils.general import set_ml_seeds, create_folder, save_dict_to_binary, load_dict_from_binary
from src.utils.metrics import compute_losses_for_dataset, compute_average_and_std
from src.plot.activation_map import (   
                                    save_acti_map_fig,                         
                                    create_tetra_links_and_reference_images,
                                    find_min_max_activation_point,
                                    find_min_activation_point_k_means,
                                    to_tetrahedral)
from src.dataset.inria import ( 
                        load_inria_patient_electrodes_mesh, 
                        get_inria_small_mesh_path, 
                        get_inria_sample_path, 
                        get_geometry_info_per_patient_dict, 
                        get_masks_per_patient_dict)
from src.utils.nn import get_parameters_count_from_model
from src.utils.math import compute_closest_points_combination

np.set_printoptions(
    edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x)
)
logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"


@hydra.main(version_base="1.3", config_path="conf", config_name="ecgi-local")
def my_app(cfg) -> None:
    print(tf.__version__)
    set_ml_seeds(cfg.seed)
    print(cfg)

    # Save configuration file
    create_folder(cfg.model.tb_output_dir)
    filename = os.path.join(cfg.model.tb_output_dir,"conf.yaml")
    with open(filename,"w") as fp:
        OmegaConf.save(config=cfg, f=fp)
    #Image saving function
    def save_png_eps_figure(filename):
        if(cfg.save_plots):
            full_filename = os.path.join(cfg.model.tb_output_dir,filename)
            plt.savefig(full_filename+".png")
            # plt.savefig(full_filename+".eps")
            plt.savefig(full_filename+".pdf")
    # Dict saving function
    def save_dict(in_dict, filename):
        if(cfg.save_plots):
            full_filename = os.path.join(cfg.model.tb_output_dir, filename)
            with open(full_filename, "w") as f:
                json.dump(in_dict, f)
    def load_dict(filename):
        full_filename = os.path.join(cfg.model.tb_output_dir, filename)
        with open(full_filename, "r") as f:
            return json.load(f)
    #Numpy saving function
    def save_np(np_array, filename):
        if(cfg.save_plots):
            full_filename = os.path.join(cfg.model.tb_output_dir,filename)
            np.save(full_filename, np_array)
    # Pickle saving function
    def save_pickle(in_pickle, filename):   
        full_filename = os.path.join(cfg.model.tb_output_dir,filename)   
        with open(full_filename, 'wb') as file:
            pickle.dump(in_pickle, file)
    def load_pickle(filename):
        full_filename = os.path.join(cfg.model.tb_output_dir, filename)
        with open(full_filename, 'rb') as file:
            return pickle.load(file)
    # Plot settings
    labelssize = 11
    ticksize = 11
    legendsize = 12

    # %%
    # Load dataset
    train_ds, val_ds, test_ds = call(cfg.dataset.load_function)
    train_ds_len = len(train_ds)
    val_ds_len = len(val_ds)
    test_ds_len = len(test_ds)

    # %%
    # Instantiate model
    model = instantiate(cfg.model)
    #Instantiate optimizer and compile the model
    opt = instantiate(cfg.optimizer)
    model.compile(optimizer=opt)
    # Instantiate callbacks
    callbacks = [instantiate(cfg.callbacks[c]) for c in cfg.callbacks]
    training_time_cb_index = list(cfg.callbacks.keys()).index("training-time")
    outs = model({"acti_map":tf.zeros((cfg.batch_size,120,120,120)),
                "conduct":tf.zeros((cfg.batch_size,120,120,120)),
                "mask":tf.zeros((cfg.batch_size,120,120,120)),
                "signal":tf.zeros((cfg.batch_size,260,450)),
                "adj_matrix":tf.zeros((cfg.batch_size,260,260)),
                "patient_id":tf.zeros((cfg.batch_size,)),
                "peaks_location":tf.zeros((cfg.batch_size,2))},
                training=False)
    # Test if evaluate and predict work
    # model.evaluate(train_ds.batch(cfg.batch_size).take(1))
    # model.predict(train_ds.batch(cfg.batch_size).take(1))
    # model.summary(expand_nested=True)

    # %%
    # Train model and collect history and other results
    start_train_time = time.time()
    history = model.fit(train_ds.batch(cfg.batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE), epochs=cfg.train.num_epochs, 
                        validation_data=val_ds.batch(cfg.batch_size,drop_remainder=True).prefetch(tf.data.AUTOTUNE), callbacks=callbacks)
    end_train_time = time.time()
    train_time = end_train_time-start_train_time
    train_epoch_training_times = callbacks[training_time_cb_index].epoch_times
    save_dict({"train_time":train_time,
            "train_epochs_time":train_epoch_training_times,
            },"train_time")

    # %%
    checkpoint_path = os.path.join(cfg.model.tb_output_dir,"checkpoint") #Restore best model 
    def reload_best_model_with_leads(num_removed_leads:int=0):
        # Load best model from checkpoint and sets the number of employed leads if specified
        # Load best model from checkpoint
        model = instantiate(cfg.model)
        model.load_weights(checkpoint_path).expect_partial()
        # Set number of leads if specified
        if(num_removed_leads>0):
            model.num_kept_leads = model.num_leads - num_removed_leads
        # Build the model by calling it on an example data
        # model.predict(train_ds.batch(cfg.batch_size).take(1))
        model({"acti_map":tf.zeros((cfg.batch_size,120,120,120)),
                    "conduct":tf.zeros((cfg.batch_size,120,120,120)),
                    "mask":tf.zeros((cfg.batch_size,120,120,120)),
                    "signal":tf.random.uniform((cfg.batch_size,260,450)),
                    "adj_matrix":tf.zeros((cfg.batch_size,260,260)),
                    "peaks_location":tf.zeros((cfg.batch_size,2)),
                    "patient_id":tf.zeros((cfg.batch_size,))},
                    training=False)
        # Compile the model
        opt = instantiate(cfg.optimizer)
        model.compile(optimizer=opt)
        return model
    model = reload_best_model_with_leads()

    # %%
    if(cfg.save_plots):
        if(cfg.model.name in ["latent-lead"]):
            if(cfg.model.gnn.name!="uniform"):
                # Evaluate model for different number of employed leads
                removed_leads_range = range(0, 250, 20) #take(1)
                for num_removed_leads in removed_leads_range:
                    model = reload_best_model_with_leads(num_removed_leads)
                    # Compute losses of final model on training and test sets
                    train_loss_dict = model.evaluate(train_ds.batch(cfg.batch_size, drop_remainder=True),return_dict=True) #.take(1)
                    filename = os.path.join(cfg.model.tb_output_dir,f"train_loss_{num_removed_leads}")
                    save_dict(train_loss_dict,filename)
                    val_loss_dict = model.evaluate(val_ds.batch(cfg.batch_size, drop_remainder=True),return_dict=True)#.take(1)
                    filename = os.path.join(cfg.model.tb_output_dir,f"val_loss_{num_removed_leads}")
                    save_dict(val_loss_dict,filename)
                    test_loss_dict = model.evaluate(test_ds.batch(cfg.batch_size, drop_remainder=True),return_dict=True)#.take(1)
                    filename = os.path.join(cfg.model.tb_output_dir,f"test_loss_{num_removed_leads}")
                    save_dict(test_loss_dict,filename)
    # Plot test losses from previous cell results (different number of input leads)
    if(cfg.save_plots):
        if(cfg.model.name in ["latent-lead"]):
            if(cfg.model.gnn.name!="uniform"):
                results_less_leads_dict = {key:np.zeros(len(removed_leads_range),dtype=float) for key in test_loss_dict.keys()}
                results_less_leads_dict["num_removed_leads"] = np.zeros(len(removed_leads_range))
                for l_i, num_removed_leads in enumerate(removed_leads_range):
                    filename = os.path.join(cfg.model.tb_output_dir,f"test_loss_{num_removed_leads}")
                    test_loss_dict = load_dict(filename)
                    for key in test_loss_dict.keys():
                        key_array = results_less_leads_dict[key]
                        key_array[l_i] = test_loss_dict[key]
                        results_less_leads_dict[key] =  key_array
                    num_removed_leads_array = results_less_leads_dict["num_removed_leads"]
                    num_removed_leads_array[l_i] = num_removed_leads
                    results_less_leads_dict["num_removed_leads"] =  num_removed_leads_array
                map_plot_dict = {'loss': "Loss",
                'acti_loss': "Activation Loss",
                'signal_loss': "Signal Loss",
                'mae_acti_loss': "MAE Activation Loss",
                'alpha_loss': "FMM localization error"}
                for key in test_loss_dict.keys():
                    plt.figure()
                    plt.plot(results_less_leads_dict["num_removed_leads"],results_less_leads_dict[key])
                    plt.xlabel("Number of removed leads", fontsize=labelssize)
                    plt.ylabel(map_plot_dict[key], fontsize=labelssize)
                    plt.xticks(fontsize=ticksize)
                    plt.yticks(fontsize=ticksize)
                    plt.legend(fontsize=legendsize)
                    save_png_eps_figure(f"{key}_vs_removed_leads")
                    plt.close()

    # %%
    # Reload full model
    model = reload_best_model_with_leads()
    # Compute and save model size
    model_parameters_count_dict = get_parameters_count_from_model(model)
    model_size_bytes = os.path.getsize(checkpoint_path+".data-00000-of-00001")
    model_size_mb = model_size_bytes / (1024 * 1024)
    model_size_dict = {"num_trainable": model_parameters_count_dict["num_trainable"],
                        "num_non_trainable": model_parameters_count_dict["num_non_trainable"],
                        "num_parameters": model_parameters_count_dict["num_parameters"],
                        "model_size_bytes": model_size_bytes,
                        "model_size_mb": model_size_mb,
                        }
    if(cfg.model.name in ["latent-lead"]):
        for submodel,submodel_name in zip([model.gnn, model.encoder, model.decoder, model.fmm_head],
                                        ["gnn", "encoder", "decoder", "fmm_head"]):
            submodel_parameters_count_dict = get_parameters_count_from_model(submodel)
            model_size_dict[submodel_name] = submodel_parameters_count_dict
    if(cfg.save_plots):
        save_dict(model_size_dict,"model_size")
    print(model_size_dict)

    # %%
    if(cfg.save_plots):
        if(cfg.model.name in ["latent-lead"]):
            from src.plot.fmm import plot_fmm_wave_from_coefficients
            # Plot FMM waves coefficients and the lead weights from the gnn
            num_samples_to_plot = 5
            lead_to_plot = 100
            for ds,ds_name in zip([test_ds,train_ds],["test","train"]):
                for sample in ds.shuffle(100,seed=cfg.seed).take(cfg.batch_size).batch(cfg.batch_size):
                    predict_dict = model.predict(sample)
                    patient_ids = sample["patient_id"].numpy()
                    onsets = sample["onset"].numpy()
                    file_ids = sample["file_id"].numpy()
                    for i in range(num_samples_to_plot):
                        patient_id = patient_ids[i].decode('utf-8')
                        onset = onsets[i].decode('utf-8')
                        file_id = file_ids[i].decode('utf-8')
                        original = predict_dict["signal"][i]
                        # Extract FMM coefficients
                        fmm_coeffs = predict_dict["predicted_fmm_coefficients"][i]
                        sample_len = cfg.dataset.sequence_length
                        xaxis = np.arange(1,sample_len+1)/cfg.dataset.fs
                        # Plot original ECG time series
                        plt.figure()
                        plt.plot(xaxis,np.squeeze(original[:,lead_to_plot]),label="Original",color="b")
                        plt.xlabel("Time [s]",fontsize=labelssize)
                        plt.ylabel("ECG",fontsize=labelssize)
                        plt.xticks(fontsize=ticksize)
                        plt.yticks(fontsize=ticksize)
                        # Plot ECG data
                        plot_fmm_wave_from_coefficients(fmm_coeff_array=fmm_coeffs,
                                                        num_leads=cfg.dataset.num_electrodes,
                                                        num_waves = cfg.model.num_waves,
                                                        seq_len=cfg.dataset.sequence_length,
                                                        fs=cfg.dataset.fs,
                                                        lead=lead_to_plot,
                                                        add_single_waves=True,
                                                        label="Predicted",
                                                        wave_label_type="number")
                        plt.legend(fontsize=legendsize)
                        save_png_eps_figure(f"fmm_reconstruction_{ds_name}_{patient_id}_{onset}_{file_id}_lead_{lead_to_plot}")
                        plt.close()
                        # Plot the lead weights from the gnn
                        # Get electrodes mesh from data folder for the correct patient
                        electrode_mesh = load_inria_patient_electrodes_mesh(cfg.datapath,patient_id)
                        # Get output weights from the gnn
                        lead_weight_sample = predict_dict["lead_weights"][i]
                        # Do the scatter plot of the weights
                        fig = plt.figure()
                        ax = fig.add_subplot(projection='3d')
                        sc = ax.scatter(electrode_mesh[:, 0], electrode_mesh[:, 1], electrode_mesh[:, 2], c=lead_weight_sample)
                        ax.set_xlabel("x", fontsize=labelssize)
                        ax.set_ylabel("y", fontsize=labelssize)
                        ax.set_zlabel("z", fontsize=labelssize)
                        ax.tick_params(axis='both', which='major', labelsize=ticksize)
                        ax.tick_params(axis='z', which='major', labelsize=ticksize)
                        ax.view_init(elev=20, azim=145)
                        fig.colorbar(sc)
                        save_png_eps_figure(f"gnn_lead_weights_{ds_name}_{patient_id}_{onset}_{file_id}")
                        plt.close()

    # %%
    # Plot training history
    def history_plot_fun(history, file_name):
        plt.figure()
        plt.rcParams.update({'font.size': 8})
        if(cfg.model.name=="vae-inria"):
            plot_list = [['loss','val_loss'],['rec_loss','val_rec_loss'], ['rec_loss_gaus','val_rec_loss_gaus'],['kl_loss','val_kl_loss']]
        elif(cfg.model.name=="latent-lead"):
            plot_list = [['loss','val_loss'],['acti_loss','val_acti_loss'],['signal_loss','val_signal_loss'], ['alpha_loss','val_alpha_loss']]
        plot_list.append(['lr'])
        for i,metric_to_plot_list in enumerate(plot_list):
            ax = plt.subplot(len(plot_list),1,i+1)
            for metric_to_plot in metric_to_plot_list:
                to_plot = history.history[metric_to_plot]
                ax.plot(to_plot,label=metric_to_plot)
            plt.legend()
            ax.set_title(metric_to_plot_list[0].capitalize())
            ax.set_ylabel("Loss")
            if(i!=len(plot_list)-1):
                ax.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False)
        ax.set_xlabel("Epoch")
        if(cfg.save_plots):
            filename = os.path.join(cfg.model.tb_output_dir, f"{file_name}.json")
            with open(filename, "w") as f:
                json.dump(history.history, f)
            save_png_eps_figure(f"{file_name}")
    if(history is not None):
        history_plot_fun(history=history, file_name="history")

    # %%
    if(cfg.save_plots):
        if(cfg.dataset.name=="inria"):
            num_samples_to_plot = 5
            # for ds,slice in zip([test_ds,train_ds],["test","train"]):
            for ds,slice in zip([test_ds],["test"]):
                # for sample in ds.shuffle(100,seed=cfg.seed).take(cfg.batch_size).batch(cfg.batch_size):
                for sample in ds.take(cfg.batch_size).batch(cfg.batch_size):
                    predict_dict = model.predict(sample)
                    patient_ids = sample["patient_id"].numpy()
                    onsets = sample["onset"].numpy()
                    file_ids = sample["file_id"].numpy()
                    # Collect groundtruth, predicted activation map and mask for the sample batch 
                    ground_truth_acti_map_tens = tf.sparse.to_dense(predict_dict["acti_map"])
                    predicted_acti_map_tens = predict_dict["predicted_acti_map"]
                    mask_tens = tf.cast(tf.sparse.to_dense(predict_dict["mask"]),tf.bool)
                    # Mask activation maps with the extracted mask
                    ground_truth_acti_map_tens_masked = tf.where(mask_tens, ground_truth_acti_map_tens, tf.zeros_like(ground_truth_acti_map_tens))
                    # ground_truth_acti_map_tens_masked = (
                    #     1 * tf.cast(mask_tens, ground_truth_acti_map_tens_masked.dtype) - ground_truth_acti_map_tens_masked
                    # )          
                    predicted_acti_map_tens_masked = tf.where(mask_tens, predicted_acti_map_tens, tf.zeros_like(predicted_acti_map_tens))
                    #For prediction, use 1.0 - predicted_acti_map to invert as in the loss function
                    predicted_acti_map_tens_masked = (
                        1 * tf.cast(mask_tens, predicted_acti_map_tens_masked.dtype) - predicted_acti_map_tens_masked
                    )                         
                    for i in range(num_samples_to_plot):
                        ground_truth_acti_map = ground_truth_acti_map_tens_masked.numpy()[i]
                        predicted_acti_map = predicted_acti_map_tens_masked.numpy()[i]
                        try:
                            # Save the predicted activation map figure
                            save_acti_map_fig(acti_map=predicted_acti_map, 
                                            data_path=cfg.datapath,
                                            save_path=cfg.model.tb_output_dir,
                                            slice=slice,
                                            patient_id=patient_ids[i].decode('utf-8'),
                                            onset=onsets[i].decode('utf-8'),
                                            file_id=file_ids[i].decode('utf-8'),
                                            add_text=slice+"_pred")
                            # Save the real activation map figure
                            save_acti_map_fig(acti_map=ground_truth_acti_map,
                                            data_path=cfg.datapath,
                                            save_path=cfg.model.tb_output_dir,
                                            slice=slice,
                                            patient_id=patient_ids[i].decode('utf-8'),
                                            onset=onsets[i].decode('utf-8'),
                                            file_id=file_ids[i].decode('utf-8'),
                                            add_text=slice+"_real")
                            # # Plot the abs difference of the predicted and real activation map
                            # save_acti_map_fig(acti_map=np.abs(ground_truth_acti_map-predicted_acti_map),
                            #                 data_path=cfg.datapath,
                            #                 save_path=cfg.model.tb_output_dir,
                            #                 slice=slice,
                            #                 patient_id=patient_ids[i].decode('utf-8'),
                            #                 onset=onsets[i].decode('utf-8'),
                            #                 file_id=file_ids[i].decode('utf-8'),
                            #                 add_text=slice+"_difference")   
                        except:
                            a=0

    # %%
    def compute_onset_position(in_dataset, mask_per_patient_dict, tetra_links_dict, mesh_dict, reference_images_dict, geometry_info_dict, transformer):
        """Computes onest position for predicted and real activation map and their distance 

        Args:
            in_dataset (_type_): dataset
            other_inputs: in order to have speed up, some structures are presaved and given as input
        Outs:
            a list of dictionaries containing patient_id, onset, file_id, real onset position, predicted onset position, their distance 

        """
        
        results_list = []
        for sample in tqdm.tqdm(in_dataset.batch(cfg.batch_size, drop_remainder=True)): #.take(1)
            predict_dict = model.predict(sample,verbose=0)
            patient_ids = sample["patient_id"].numpy()
            onsets = sample["onset"].numpy()
            file_ids = sample["file_id"].numpy()


            # Collect groundtruth, predicted activation map and mask for the sample batch 
            ground_truth_acti_map_tens = tf.sparse.to_dense(predict_dict["acti_map"])
            predicted_acti_map_tens = predict_dict["predicted_acti_map"]
            mask_tens = tf.cast(tf.sparse.to_dense(predict_dict["mask"]),tf.bool)
            # Mask activation maps with the extracted mask
            ground_truth_acti_map_tens_masked = tf.where(mask_tens, ground_truth_acti_map_tens, tf.zeros_like(ground_truth_acti_map_tens))
            # ground_truth_acti_map_tens_masked = (
            #     1 * tf.cast(mask_tens, ground_truth_acti_map_tens_masked.dtype) - ground_truth_acti_map_tens_masked
            # )          
            predicted_acti_map_tens_masked = tf.where(mask_tens, predicted_acti_map_tens, tf.zeros_like(predicted_acti_map_tens))
            #For prediction, use 1.0 - predicted_acti_map to invert as in the loss function
            predicted_acti_map_tens_masked = (
                1 * tf.cast(mask_tens, predicted_acti_map_tens_masked.dtype) - predicted_acti_map_tens_masked
            )                         
            for i in range(cfg.batch_size):
                patient_id=patient_ids[i].decode('utf-8')
                onset=onsets[i].decode('utf-8')
                file_id=file_ids[i].decode('utf-8')
                ground_truth_acti_map = ground_truth_acti_map_tens_masked.numpy()[i]
                predicted_acti_map = predicted_acti_map_tens_masked.numpy()[i]
                num_onsets = {"one_init_rv":1,"two_init_lv":2,"three_init":3}.get(onset)
                # ground_truth_acti_map = tf.sparse.to_dense(predict_dict["acti_map"]).numpy()[i] # Get real activation map
                onsets_positions, cluster_centers = find_min_activation_point_k_means(acti_map=ground_truth_acti_map,
                                                    num_onsets = num_onsets,
                                                    acti_mask=mask_per_patient_dict[patient_id],
                                                    tetra_links=tetra_links_dict[patient_id], 
                                                    tetra_polydata=mesh_dict[patient_id],
                                                    ref_img=reference_images_dict[patient_id],
                                                    geometry_info=geometry_info_dict[patient_id], 
                                                    transformer=transformer,
                                                )
                #For prediction, use 1.0 - predicted_mask to invert as in the loss function
                # predicted_acti_map = 1.0 - predict_dict["predicted_acti_map"][i]
                onsets_positions_pred, cluster_centers_pred = find_min_activation_point_k_means(acti_map=predicted_acti_map,
                                                    num_onsets = num_onsets,
                                                    acti_mask=mask_per_patient_dict[patient_id],
                                                    tetra_links=tetra_links_dict[patient_id], 
                                                    tetra_polydata=mesh_dict[patient_id],
                                                    ref_img=reference_images_dict[patient_id],
                                                    geometry_info=geometry_info_dict[patient_id], 
                                                    transformer=transformer,
                                                )
                total_distance, onsets_positions_ordered, onsets_positions_pred_ordered = \
                    compute_closest_points_combination(onsets_positions, onsets_positions_pred)
                total_distance_clusters, clusters_positions_ordered, clusters_positions_pred_ordered = \
                                    compute_closest_points_combination(cluster_centers, cluster_centers_pred)
                sample_out_dict = {"patient_id":patient_id, "onset":onset,"num_onsets":num_onsets, "file_id":file_id,
                                "onset_position":onsets_positions_ordered, "onset_position_pred":onsets_positions_pred_ordered, 
                                "avg_distance":total_distance/num_onsets, "cluster_position":clusters_positions_ordered,
                                "cluster_position_pred":clusters_positions_pred_ordered, "avg_distance_clusters":total_distance_clusters/num_onsets}
                results_list.append(sample_out_dict)
        return results_list
        

    # %%
    # Compute onset distance on the test set for different number of employed leads
    if(cfg.save_plots):
        # Initialize structures to perform activation point 
        patient_ids_list = [f"Patient{num:03}" for num in [4, 5, 11, 12, 13, 14, 15]]
        mesh_dict = {patient_id:to_tetrahedral(pv.read(
            get_inria_small_mesh_path(datapath=cfg.datapath, patient_id=patient_id)
        )) for patient_id in patient_ids_list}
        geometry_info_dict = get_geometry_info_per_patient_dict(cfg.datapath) # Get geometry information per patient
        mask_per_patient_dict = get_masks_per_patient_dict(cfg.datapath) # Get mask for each patient
        tetra_links_dict, reference_images_dict = create_tetra_links_and_reference_images(cfg.datapath) #Get surface links and reference images per patient
        transformer = sitk.Transform()
        transformer.SetIdentity()

        if(cfg.model.name in ["latent-lead"]):
            if(cfg.model.gnn.name!="uniform"):
                # Evaluate model for different number of employed leads
                removed_leads_range = range(0, 260, 50) #.take(1)
                for num_removed_leads in removed_leads_range:
                    model = reload_best_model_with_leads(num_removed_leads)
                    # Compute losses of final model on training and test sets
                    onsets_dict_list = compute_onset_position(in_dataset=test_ds,
                                                                mask_per_patient_dict=mask_per_patient_dict,
                                                                tetra_links_dict=tetra_links_dict,
                                                                mesh_dict=mesh_dict,
                                                                reference_images_dict=reference_images_dict,
                                                                geometry_info_dict=geometry_info_dict,
                                                                transformer=transformer)

                    # Save aggregated statistics
                    onset_distances = [d['avg_distance'] for d in onsets_dict_list] # Get onset distances 
                    filename = os.path.join(cfg.model.tb_output_dir, f"onset_d_dict_{num_removed_leads}")
                    onset_dict_to_save = {"avg_d_min": np.average(onset_distances),
                                        "std_d_min": np.std(onset_distances),
                                        } 
                    save_dict(onset_dict_to_save,filename)
                    # Plot histograms
                    plt.figure()
                    plt.hist(onset_distances)
                    plt.xlabel('Distance', fontsize=labelssize)
                    plt.ylabel('Frequency', fontsize=labelssize)
                    plt.xticks(fontsize=ticksize)
                    plt.yticks(fontsize=ticksize)
                    # plt.title(f'Histogram of Onset Distances (Removed Leads: {num_removed_leads})')
                    save_png_eps_figure(os.path.join(cfg.model.tb_output_dir, f"min_onset_d_histogram_{num_removed_leads}"))
                    plt.close()
                    # Save full list of onsets dictionary
                    save_pickle(onsets_dict_list, filename=f"onsets_dict_list_{num_removed_leads}")
                    # load with: load_pickle(filename="onsets_dict_list")
        else:
            model = reload_best_model_with_leads()
            # Compute losses of final model on training and test sets
            onsets_dict_list = compute_onset_position(in_dataset=test_ds,
                                                        mask_per_patient_dict=mask_per_patient_dict,
                                                        tetra_links_dict=tetra_links_dict,
                                                        mesh_dict=mesh_dict,
                                                        reference_images_dict=reference_images_dict,
                                                        geometry_info_dict=geometry_info_dict,
                                                        transformer=transformer)
            # Save aggregated statistics
            onset_distances = [d['avg_distance'] for d in onsets_dict_list] # Get onset distances 
            filename = os.path.join(cfg.model.tb_output_dir, f"onset_d_dict_0")
            onset_dict_to_save = {"avg_d_min": np.average(onset_distances),
                                "std_d_min": np.std(onset_distances),
                                } 
            save_dict(onset_dict_to_save,filename)
            save_pickle(onsets_dict_list, filename=f"onsets_dict_list_0")


    # %%
    # Compute losses of final model on training and test sets
    model = reload_best_model_with_leads()
    train_loss_dict = model.evaluate(train_ds.batch(cfg.batch_size, drop_remainder=True),return_dict=True)
    filename = os.path.join(cfg.model.tb_output_dir,"train_loss")
    save_dict(train_loss_dict,filename)
    val_loss_dict = model.evaluate(val_ds.batch(cfg.batch_size, drop_remainder=True),return_dict=True)
    filename = os.path.join(cfg.model.tb_output_dir,"val_loss")
    save_dict(val_loss_dict,filename)
    test_loss_dict = model.evaluate(test_ds.batch(cfg.batch_size, drop_remainder=True),return_dict=True)
    filename = os.path.join(cfg.model.tb_output_dir,"test_loss")
    save_dict(test_loss_dict,filename)

    if cfg.model.name in ["latent-lead"]:
        loss_keys = ["total_loss", "acti_loss", "signal_loss", "mae_acti_loss", "alpha_loss"]
    elif cfg.model.name in ["vae-inria"]:
        loss_keys = ["total_loss", "rec_loss", "rec_loss_gaus", "kl_loss", "rec_loss_mae"]

    test_loss_stats = compute_average_and_std(test_ds, model, batch_size=cfg.batch_size, metrics=loss_keys)
    save_pickle(test_loss_stats, filename="test_loss_stats")

    # %%
    # Set completed in configuration file and save it in the experiment folder
    cfg.completed = True
    filename = os.path.join(cfg.model.tb_output_dir,"conf.yaml")
    with open(filename,"w") as fp:
        OmegaConf.save(config=cfg, f=fp)
    print("Script ends")


if __name__ == "__main__":
    my_app()
