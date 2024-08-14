<!-- # Code for:
Reducing the number of leads for ECG Imaging with Graph Neural Networks and meaningful latent space
This work was supported by the Swedish Research Council (project "Scalable Federated Learning", no. 2021-04610) and by the King Abdullah University of Science and Technology (KAUST) Office of Research Administration (ORA) under Award No. ORA-CRG2021-4699. Daniel F. Perez-Ramirez is supported by the Swedish Foundation for Strategic Research (SSF). We acknowledge Euro HPC Joint Undertaking for awarding us access to the Leonardo supercomputer located at the CINECA data center in Bologna, Italy. -->

## ECG Imaging

This repository contains the code for the paper "Reducing the number of leads for ECG Imaging with Graph Neural Networks and meaningful latent space" paper [1]. Please [cite our paper](#citation) if you use our code.

### Download the dataset
Please download the dataset by asking the authors from [2]. Your data directory should look like: 

        data/
        │
        └───simulation/
                │
                ├───mesh_sources/
                │   └───PatientXXX/
                │       ├───electrodes_mesh.vtk
                │       └───small_mesh.vtk
                │
                ├───test/
                │   └───PatientXXX/
                │       ├───one_init_rv/
                │       ├───two_init_lv/
                │       ├───three_init/
                │
                └───train/
                        └───PatientXXX/
                        ├───one_init_rv/
                        ├───two_init_lv/
                        ├───three_init/

### Setup environment

1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create a conda environment. Use the commands in [install.txt](install.txt) to create a new environment named "ecgi" (replace "ecgi" with your desired environment name):

        conda create -n ecgi python=3.11
        conda activate ecgi

        conda install cudatoolkit=11.8 -y
        conda install cudnn=8.9 -y

        conda install conda-forge::matplotlib -y
        conda install conda-forge::hydra-core -y
        conda install conda-forge::pandas -y

        python -m pip install tensorflow[and-cuda]==2.14
        python -m pip install SimpleITK
        python -m pip install tetgen
        python -m pip install vtk
        python -m pip install pyvista
        python -m pip install pymeshfix
        python -m pip install spektral
        python -m pip install PyWavelets


We provide 2 equivalent ways to run our code: 
1. A python script [ecgi.py](ecgi.py) 
2. A notebook [ecgi.ipynb](ecgi.ipynb) useful for interactivity

### Running in Python
Run [ecgi.py](/ecgi.py) in a terminal with datasets and models from [the following section](#configurations) and the arguments you can find in [the configuration file](/conf/ecgi-local.yaml). For instance, to run the model *latent-lead* on the *inria* dataset, learning rate 0.0001, 240 epochs, batch size 32, reconstruction_signal_weight 1.0, alpha_loss_weight=1.0, and GAT layers for the GNN:

        conda activate ecgi
        python ecgi.py dataset=inria model=latent-lead model/gnn=gat_model batch_size=32 optimizer=adam optimizer.learning_rate=0.0001 train num_epochs=240 save_plots=True model.reconstruction_signal_weight=1 0 model.alpha_loss_weight=1.0  


### Running in notebook
The main notebook is [ecgi.ipynb](/ecgi.ipynb). You can update the configuration by modifying the overrides arguments for the hydra `compose` function in the first cell:

        cfg = compose(config_name='ecgi-local.yaml',return_hydra_config=True,
                overrides=["hydra.verbose=true","dataset=inria",
                                "model=vae-inria",                  
                                #  "model=latent-lead",
                                #  "model/gnn=gat_model",
                                "batch_size=32", 
                                "optimizer=adam",
                                "optimizer.learning_rate=0.0001", 
                                "train.num_epochs=0",
                                "save_plots=True",
                                ]) 


For additional parameters, please refer to the [configuration file](/conf/ecgi-local.yaml).

### Running VAE baseline
Example command to run the vae baseline:

        python ecgi.py dataset=inria model=vae-inria batch_size=32 optimizer=adam optimizer.learning_rate=0.001 train.num_epochs=240 save_plots=True seed=42


### Collecting results
The output data, images and models are saved into the `tb_output_dir` path of the [configuration file](/conf/ecgi-local.yaml). You can collect results from runs by means of functions similar to the ones provided in the [collect_results notebook](/collect_results.ipynb).

## Citation

If you use this code in your work, please cite our paper [1]:


### References
[1]: Giacomo Verardo, Daniel F. Perez-Ramirez, Samuel Bruchfeld, Magnus Boman, Marco Chiesa, Sabine Koch, Gerald Q. Maguire Jr., Dejan Kostic. (2024). Reducing the number of leads for ECG Imaging with Graph Neural Networks and meaningful latent space. Accepted at *STACOM, at MICCAI 2024*.

[2] Bacoyannis, T., Ly, B., Cedilnik, N., Cochet, H., Sermesant, M. (2021). Deep learning formulation of electrocardiographic imaging integrating image and signal information with data-driven regularization. *EP Europace*, 23(Supplement_1), i55–i62. https://doi.org/10.1093/europace/euaa391

