import numpy as np
from src.utils.general import scalar_to_np_array
import tensorflow as tf
import math 
from typing import List, Dict

def fmm_wave(A,alpha,beta,omega,wave_len, batch_dim=None):
    A = np.reshape(A,(-1,1))
    alpha = np.reshape(alpha,(-1,1))
    beta = np.reshape(beta,(-1,1))
    omega = np.reshape(omega,(-1,1))
    t = np.linspace(0,2*np.pi,wave_len)
    if(batch_dim):
        t = np.tile(t,reps=[batch_dim,1])
    # Phase: ϕ (t; α, β, ω) = β + 2 arctan ω tan((t - α)/2)
    phase = beta + 2*np.arctan(omega*np.tan((t-alpha)/2))
    # Wave: A cos (ϕ (t; α, β, ω)) 
    wave = A*np.cos(phase)
    return wave

def fmm_wave_tf(A,alpha,beta,omega,wave_len):
    A = tf.reshape(A,(-1,1))
    alpha = tf.reshape(alpha,(-1,1))
    beta = tf.reshape(beta,(-1,1))
    omega = tf.reshape(omega,(-1,1))
    batch_size = tf.shape(A)[0]
    # Compute linear space between 0 and 2pi
    start = tf.constant(0, dtype=tf.float32)
    stop = tf.constant(2 * np.pi, dtype=tf.float32)
    t = tf.linspace(start, stop, wave_len)
    # Replicate the linspace tensor along the first axis for the batch dimension
    t = tf.tile(tf.expand_dims(t, 0), [batch_size, 1])
    # Phase: ϕ (t; α, β, ω) = β + 2 arctan ω tan((t - α)/2)
    phase = beta + 2*tf.math.atan(omega*tf.math.tan((t-alpha)/2))
    # Wave: A cos (ϕ (t; α, β, ω)) 
    wave = A*tf.math.cos(phase)
    return wave

def fmm_wave_tf_different_lens(A,alpha,beta,omega,wave_lengths,final_len):
    A = tf.reshape(A,(-1,1))
    alpha = tf.reshape(alpha,(-1,1))
    beta = tf.reshape(beta,(-1,1))
    omega = tf.reshape(omega,(-1,1))
    max_len = tf.reduce_max(wave_lengths)
    indices = tf.expand_dims(tf.range(max_len), axis=0)
    steps = tf.expand_dims(2 * math.pi / tf.cast(wave_lengths, tf.float32), axis=1)
    mask = tf.cast(indices < tf.expand_dims(wave_lengths, axis=1), tf.float32)
    t = tf.multiply(tf.cast(indices,float), steps) * mask
    t = t * tf.cast(mask, tf.float32)
    # Phase: ϕ (t; α, β, ω) = β + 2 arctan ω tan((t - α)/2)
    phase = beta + 2*tf.math.atan(omega*tf.math.tan((t-alpha)/2))
    # Wave: A cos (ϕ (t; α, β, ω)) 
    wave = A*tf.math.cos(phase)
    wave = wave*mask
    wave = tf.pad(wave, [[0, 0], [0, final_len - max_len]])
    return wave


def get_waves_from_fmm_model(model,dataset,cfg):
    num_batches = len(dataset)
    batch_size = cfg.batch_size
    waves = np.zeros((num_batches,batch_size,cfg.dataset.sequence_length,cfg.dataset.num_features,5))
    # m_parameter = np.zeros((num_batches,cfg.batch_size,cfg.dataset.num_features))
    num_total_coefficient_per_sample,_ = get_fmm_num_parameters(num_leads=cfg.dataset.num_features,num_waves=cfg.model.num_waves)
    fmm_coeff_matrix = np.zeros((num_batches*cfg.batch_size,num_total_coefficient_per_sample))
    for batch_index,data in enumerate(dataset):
        if(model.encoder_input_type=="tensor"):
            x = data["inputs"]
        elif(model.encoder_input_type=="dict"):
            x = data
        if(model.split_ecg):
            sizes = data["sizes"]
        elif(not(model.split_ecg)):
            sizes = None
        x = model.encoder(x)  
        if(isinstance(x,Dict)):
            x = x["output"]
        x = model.global_avg_pooling(x)
        fmm_head_output_dict = model.fmm_head(x,x_len=sizes,return_parameters_array=True,return_parameters_array_scaled=True,return_parameters_dict=True)
        fmm_coeff_array = fmm_head_output_dict["parameters_array"]
        encoded_parameters = fmm_head_output_dict["parameters_dict"]
        fmm_coeff_matrix[batch_index*batch_size:(batch_index+1)*batch_size,:] = fmm_coeff_array
        for i in range(cfg.dataset.num_features):
            for j,w in enumerate(get_wave_names(num_waves=cfg.model.num_waves)):
                if(sizes is not None):
                    wave = model.fmm_head.get_wave(parameters_dict=encoded_parameters,wave_name=w,lead=i,seq_len=sizes)
                else:
                    wave = model.fmm_head.get_wave(parameters_dict=encoded_parameters,wave_name=w,lead=i,seq_len=cfg.dataset.sequence_length)
                if(wave.ndim!=waves[batch_index,:,:,i,j].ndim):
                    wave = np.expand_dims(wave,-1)
                waves[batch_index,:,:,i,j]= wave
    waves = np.reshape(waves,(num_batches*cfg.batch_size,cfg.dataset.sequence_length,cfg.dataset.num_features,5))
    return waves,fmm_coeff_matrix

def generate_wave(parameters_dict, wave_name, lead, seq_len):
    a = parameters_dict[wave_name]["A"]
    alpha = parameters_dict[wave_name]["alpha"]
    beta = parameters_dict[wave_name]["beta"]
    omega = parameters_dict[wave_name]["omega"]

    num_leads = np.shape(a)[0]
    alpha = scalar_to_np_array(alpha,num_leads)
    beta = scalar_to_np_array(beta,num_leads)
    omega = scalar_to_np_array(omega,num_leads)

    # a = np.reshape(a,(1,num_leads))
    # alpha = np.reshape(alpha,(1,num_leads))
    # beta = np.reshape(beta,(1,num_leads))
    # omega = np.reshape(omega,(1,num_leads))
  
    wave_i = fmm_wave(A=a[lead],alpha=alpha[lead],beta=beta[lead],omega=omega[lead],wave_len=seq_len)
    # elif(split_ecg):
    #     wave_i = fmm_wave_tf_different_lens(A=a[:,lead],alpha=alpha[:,lead],beta=beta[:,lead],omega=omega[:,lead],wave_lengths=seq_len,final_len=seq_len)
    return wave_i

def generate_wave_tf(parameters_dict, wave_name, lead, seq_len, split_ecg=False):
    a = parameters_dict[wave_name]["A"]
    alpha = parameters_dict[wave_name]["alpha"]
    beta = parameters_dict[wave_name]["beta"]
    omega = parameters_dict[wave_name]["omega"]
    wave_i = fmm_wave_tf(A=a[:,lead],alpha=alpha,beta=beta[:,lead],omega=omega,wave_len=seq_len)
    return wave_i

def format_FMM_wave_coefficients(fmm_parameters):
    # Format wave parameters from list of r2py Datframes (1 for each lead)
    # to a dictionary format (see variable parameters_dict below)
    parameters_dict = {}
    num_leads = len(fmm_parameters)
    alpha = np.zeros((5,num_leads))
    beta = np.zeros((5,num_leads))
    a = np.zeros((5,num_leads))
    omega = np.zeros((5,num_leads))
    var = np.zeros((5,num_leads))
    m = np.zeros((5,num_leads))
    wave_name_list = get_wave_names(num_waves=num_waves) #["P","Q","R","S","T"] #["R","T","P","S","Q"]
    for lead_index in range(num_leads):
        # print(lead_index,type(lead_index))
        beat_fmm_parameters_df_r = fmm_parameters[lead_index]
        with (ro.default_converter + pandas2ri.converter).context(): # Convert to pandas dataframe
            beat_fmm_parameters_df_r:pd.DataFrame = ro.conversion.get_conversion().rpy2py(beat_fmm_parameters_df_r)
        for j,w in enumerate(wave_name_list):
            alpha[j,lead_index] = beat_fmm_parameters_df_r["Alpha"].loc[w] 
            beta[j,lead_index] = beat_fmm_parameters_df_r["Beta"].loc[w]
            a[j,lead_index] = beat_fmm_parameters_df_r["A"].loc[w]
            omega[j,lead_index] = beat_fmm_parameters_df_r["Omega"].loc[w]
            var[j,lead_index] = beat_fmm_parameters_df_r["Var"].loc[w]
            m[j,lead_index] = beat_fmm_parameters_df_r["M"].loc[w]
    for j,w in enumerate(wave_name_list):
        parameters_dict[w] = {"A":a[j],"alpha":alpha[j],"beta":beta[j],"omega":omega[j],"M":m[j],"var":var[j]}
    return parameters_dict

def get_fmm_num_parameters(num_leads:int, num_waves:int)->List[int]:
    """Generate number of parameters and number of parameters per wave 
    for an fmm model of an ecg with num_leads number of leads and num_waves waves

    Args:
        num_leads (int): number of leads
        num_waves (int, optional): number of waves. It is usually 5 (P,Q,R,S,T).

    Returns:
        List[int,int]: number of total parameters, number of parameters per wave
    """
    num_parameters_per_wave = (num_leads*2 + 2) 
    num_parameters = num_parameters_per_wave*num_waves + num_leads #Add also n parameters for parameter M
    return num_parameters, num_parameters_per_wave

def get_fmm_num_parameters_circular(num_leads:int, num_waves:int)->List[int]:
    num_parameters_per_wave = (num_leads +num_leads*2 + 2 + 1) # For alpha and beta we have 2x parameters
    num_parameters = num_parameters_per_wave*num_waves + num_leads #Add also n parameters for parameter M
    return num_parameters, num_parameters_per_wave

def get_num_leads_from_num_parameters(num_parameters:int, num_waves:int)->int:
    # Get number of leads from number of total fmm parameters and number of waves
    # Inverse of get_fmm_num_parameters
    return (num_parameters - 2 * num_waves)/(2*num_waves + 1)

def get_num_leads_from_num_parameters_circular(num_parameters:int, num_waves:int)->int:
    # Get number of leads from number of total fmm parameters and number of waves
    # Inverse of get_fmm_num_parameters
    raise NotImplementedError

def get_A_indexes(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters_per_wave * wave_index
    end_index = start_index + num_leads
    return start_index, end_index

def get_A_indexes_circular(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters_per_wave * wave_index
    end_index = start_index + num_leads
    return start_index, end_index

def get_alpha_indexes_circular(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters_per_wave * wave_index + num_leads
    end_index = start_index + 2
    return start_index, end_index

def get_beta_indexes_circular(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters_per_wave * wave_index + num_leads + 2
    end_index = start_index + num_leads*2
    return start_index, end_index

def get_omega_indexes_circular(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters_per_wave * wave_index + num_leads + 2 + num_leads*2
    end_index = start_index + 1
    return start_index, end_index

def get_M_indexes_circular(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters - num_leads
    end_index = num_parameters
    return start_index, end_index

def get_lead_indexes_circular(lead_index:int, num_leads:int, num_waves:int, include_shared:bool)->List:
    """Get indexes from FMM data structure

    Args:
        lead_index (int): index of the lead to be extracted
        num_leads (int): number of total leads
        num_waves (int): number of total waves
        include_shared (bool): include or not the alpha and omega shared FMM coefficients

    Returns:
        List: list of parameters indexes correspondent to lead with index lead_index
    """
    # assert include_shared==False # Can be removed since the functionality is implemented
    # Order: [nxA,1x2xalpha,nx2xbeta,1xomega]_wave_i where i is {P, Q, R, S, T } in order, then nxM
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    num_parameters_per_lead = 6*num_waves+1 if include_shared else 3*num_waves+1
    lead_indexes = []
    if(include_shared):
        for n_w in range(num_waves):
            A_start = n_w*num_parameters_per_wave
            lead_indexes.append(A_start + lead_index) # "A" parameter
            alpha_start = A_start + num_leads
            lead_indexes.append(alpha_start) # "beta" parameter (2 indexes for sine and cosine)
            lead_indexes.append(alpha_start + 1)
            beta_start = alpha_start + 2
            lead_indexes.append(beta_start+ 2*lead_index) # "beta" parameter (2 indexes for sine and cosine)
            lead_indexes.append(beta_start+ 2*lead_index + 1)
            omega_start = beta_start + 2*num_leads
            lead_indexes.append(omega_start)
    else:
        for n_w in range(num_waves):
            A_start = n_w*num_parameters_per_wave
            lead_indexes.append(A_start + lead_index) # "A" parameter
            beta_start = A_start + num_leads + 2
            lead_indexes.append(beta_start+ 2*lead_index) # "beta" parameter (2 indexes for sine and cosine)
            lead_indexes.append(beta_start+ 2*lead_index + 1)
        lead_indexes.append(num_waves*num_parameters_per_wave + lead_index) # M index
    return lead_indexes
    
def test_get_lead_indexes_circular(num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    indexes = []
    for lead_index in range(num_leads):
        lead_indexes = get_lead_indexes_circular(lead_index=lead_index, num_leads=num_leads, num_waves=num_waves, include_shared=False)
        indexes.extend(lead_indexes)
    # Add alpha and omega indexes
    for n_w in range(num_waves):
        A_start = n_w*num_parameters_per_wave
        alpha_start = A_start + num_leads
        indexes.append(alpha_start) # "beta" parameter (2 indexes for sine and cosine)
        indexes.append(alpha_start + 1)
        beta_start = alpha_start + 2
        omega_start = beta_start + 2*num_leads
        indexes.append(omega_start)
    expected_list = list(range(num_parameters))
    indexes.sort()
    print(expected_list)
    print(indexes)
    assert expected_list==indexes
        
def get_all_lead_indexes_circular(num_leads:int, num_waves:int, concat:bool=True)->tf.Tensor:
    # Compute the indexes of parameters belonging to each lead and store them in rows of lead_index_array
    # Each row i contains the index of the parameters for lead i
    # Shared parameters are excluded from the indexes

    # Compute the number of non-shared parameters per lead by using the first lead
    num_indexes_per_lead = len(get_lead_indexes_circular(0, num_leads, num_waves, include_shared=False)) 
    # Preallocate indexes array
    lead_parameter_index_arr = np.zeros((num_leads,num_indexes_per_lead),dtype=int) 
    # Compute the parameter indexes for each lead
    for lead_index in range(num_leads): 
        parameter_index_arr = get_lead_indexes_circular(lead_index, num_leads, num_waves, include_shared=False)
        lead_parameter_index_arr[lead_index] = parameter_index_arr
    indexes_tensor = tf.convert_to_tensor(lead_parameter_index_arr) # Convert indexes to tensor
    if(concat):
        # If concat, reshape the index tensor on a single row
        indexes_tensor = tf.reshape(indexes_tensor,-1) 
    return indexes_tensor

def get_parameters_names_list_circular(num_leads:int, num_waves:int):
    #Return a list of strings containing the name of the parameters in the coefficients array
    num_total_parameters,_ = get_fmm_num_parameters_circular(num_leads=num_leads,num_waves=num_waves)
    names_list = np.empty(shape=(num_total_parameters),dtype="object")
    for wave_index,w in enumerate(get_wave_names(num_waves=num_waves)): 
        for coeff_name,f in zip(["A","alpha","beta","omega"],[get_A_indexes_circular
                    ,get_alpha_indexes_circular,get_beta_indexes_circular,get_omega_indexes_circular]):
            start_index,end_index = f(wave_index=wave_index,num_leads=num_leads,num_waves=num_waves)
            names_list[start_index:end_index]="{0}_{1}".format(coeff_name,str(wave_index))
    start_index,end_index = get_M_indexes_circular(wave_index=None,num_leads=num_leads,num_waves=num_waves)
    names_list[start_index:end_index]="{0}".format("M")
    return names_list

def get_alpha_indexes(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters_per_wave * wave_index + num_leads
    end_index = start_index + 1
    return start_index, end_index

def get_beta_indexes(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters_per_wave * wave_index + num_leads + 1
    end_index = start_index + num_leads
    return start_index, end_index

def get_omega_indexes(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters_per_wave * wave_index + 2*num_leads + 1 
    end_index = start_index + 1
    return start_index, end_index

def get_M_indexes(wave_index:int, num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    start_index = num_parameters - num_leads
    end_index = num_parameters
    return start_index, end_index

def get_circular_indexes_as_boolean_t(num_leads:int, num_waves:int):
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    circular_indexes_array = tf.zeros(shape=(num_parameters),dtype=float)
    for wave_index in range(num_waves):
        start_alpha,end_alpha = get_alpha_indexes(wave_index=wave_index,num_leads=num_leads,num_waves=num_waves)
        start_beta,end_beta = get_beta_indexes(wave_index=wave_index,num_leads=num_leads,num_waves=num_waves)
        # circular_indexes_array[start_alpha,end_alpha] = 1.0
        # circular_indexes_array[start_beta,end_beta] = 1.0
        indexes_to_update = list(range(start_alpha,end_alpha))
        circular_indexes_array = tf.tensor_scatter_nd_update(circular_indexes_array, tf.expand_dims(indexes_to_update,axis=-1), tf.ones_like(indexes_to_update,dtype=float))
        indexes_to_update = list(range(start_beta,end_beta))
        circular_indexes_array = tf.tensor_scatter_nd_update(circular_indexes_array, tf.expand_dims(indexes_to_update,axis=-1), tf.ones_like(indexes_to_update,dtype=float))
    return circular_indexes_array

def get_wave_indexes(wave_index:int, num_leads:int, num_waves:int)->List:
    wave_indexes = []
    for c,f in zip(["A","Alpha","Beta","Omega"],[get_A_indexes,get_alpha_indexes,get_beta_indexes,get_omega_indexes]):
        start_index,end_index=f(wave_index=wave_index,num_leads=num_leads,num_waves=num_waves)
        c_index_list = list(range(start_index,end_index))
        wave_indexes.extend(c_index_list)
    return wave_indexes

def get_wave_indexes_circular(wave_index:int, num_leads:int, num_waves:int)->List:
    wave_indexes = []
    for c,f in zip(["A","Alpha","Beta","Omega"],
                   [get_A_indexes_circular,get_alpha_indexes_circular,
                    get_beta_indexes_circular,get_omega_indexes_circular]):
        start_index,end_index=f(wave_index=wave_index,num_leads=num_leads,num_waves=num_waves)
        c_index_list = list(range(start_index,end_index))
        wave_indexes.extend(c_index_list)
    return wave_indexes

def get_wave_names(num_waves:int)->List[str]:
    assert (num_waves <= 5) and (num_waves > 0)
    wave_names = ["P","Q","R","S","T"]
    return wave_names[:num_waves]

def get_parameters_names_list(num_leads:int, num_waves:int):
    #Return a list of strings containing the name of the parameters in the coefficients array
    num_total_parameters, _ = get_fmm_num_parameters(num_leads=num_leads, num_waves=num_waves)
    names_list = np.empty(shape=(num_total_parameters), dtype="object")
    for wave_index,w in enumerate(get_wave_names(num_waves=num_waves)): 
        for coeff_name,f in zip(["A", "alpha", "beta", "omega"],[get_A_indexes, get_alpha_indexes, get_beta_indexes, get_omega_indexes]):
            start_index, end_index = f(wave_index=wave_index, num_leads=num_leads, num_waves=num_waves)
            for lead in range(num_leads):
                names_list[start_index+lead] = "{0}_{1}_{2}".format(coeff_name, str(wave_index), lead)
    start_index,end_index = get_M_indexes(wave_index=None, num_leads=num_leads, num_waves=num_waves)
    names_list[start_index:end_index] = "{0}".format("M")
    return names_list

def convert_batched_fmm_dictionary_to_array(fmm_dict:Dict, batch_size:int)->np.ndarray:
    num_leads=1
    num_waves = len(fmm_dict)
    # num_leads = fmm_dict["P"]["A"].shape[0] # Get number of leads from an element in the dictionary (e.g., P wave, coefficient A)
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    coefficients_array =  np.zeros((batch_size,num_parameters))
    for wave_index,w in enumerate(get_wave_names(num_waves=num_waves)):
        wave_dict = fmm_dict[w]
        for coeff_name,f in zip(["A","alpha","beta","omega"],[get_A_indexes, get_alpha_indexes,get_beta_indexes,get_omega_indexes]):
            start_index,end_index = f(wave_index=wave_index,num_leads=num_leads,num_waves=num_waves)
            for batch_index in range(batch_size):
                coefficients_array[batch_index,start_index:end_index] = np.squeeze(wave_dict[coeff_name][batch_index])
    start_index,end_index = get_M_indexes(wave_index=None,num_leads=num_leads,num_waves=num_waves)
    for batch_index in range(batch_size):
        coefficients_array[batch_index,start_index:end_index] = wave_dict["M"][batch_index]
    return coefficients_array


def convert_fmm_dictionary_to_array(fmm_dict:Dict)->np.ndarray:
    num_waves = len(fmm_dict)
    num_leads = fmm_dict["P"]["A"].shape[0] # Get number of leads from an element in the dictionary (e.g., P wave, coefficient A)
    num_parameters, num_parameters_per_wave = get_fmm_num_parameters(num_leads=num_leads,num_waves=num_waves)
    coefficients_array =  np.zeros((num_parameters))
    # for wave_index,(wave_name,wave_dict) in enumerate(fmm_dict.items()):  
    for wave_index,w in enumerate(get_wave_names(num_waves=num_waves)):
        wave_dict = fmm_dict[w]
        for coeff_name,f in zip(["A","alpha","beta","omega"],[get_A_indexes, get_alpha_indexes,get_beta_indexes,get_omega_indexes]):
            start_index,end_index = f(wave_index=wave_index,num_leads=num_leads,num_waves=num_waves)
            # coefficients_array[start_index:end_index] = wave_dict[coeff_name]
            if(coeff_name=="A" or coeff_name=="beta"): # For coefficients that have a different value for each lead
                coefficients_array[start_index:end_index] = np.squeeze(wave_dict[coeff_name]) #Add all the coefficient array to the final array
            if(coeff_name=="alpha" or coeff_name=="omega"): # For coefficients that have the same value for each lead
                wave_coeff = np.squeeze(wave_dict[coeff_name])
                np.testing.assert_equal(wave_coeff,wave_coeff[0]*np.ones_like(wave_coeff)) # Check that all the elements in array are equal
                coefficients_array[start_index:end_index] = wave_coeff[0] # Select the first one which is the same as the others
    start_index,end_index = get_M_indexes(wave_index=None,num_leads=num_leads,num_waves=num_waves)
    coefficients_array[start_index:end_index] = wave_dict["M"]
    return coefficients_array

def extract_fmm_lead_from_array(fmm_coefficients_array,lead:int,num_leads:int,num_waves:int)->np.ndarray:
    # Extact single lead from array of fmm coefficients taken for num_leads leads. Does not extract multiple leads
    num_parameters_single_lead, num_parameters_per_wave_single_lead = get_fmm_num_parameters(num_leads=1,num_waves=num_waves) #Only one lead is selected
    extracted_lead_fmm_coefficients_array = np.zeros((num_parameters_single_lead))
    assert num_waves==5
    for wave_index,w in enumerate(get_wave_names(num_waves=num_waves)):
        for coeff_name,f in zip(["A","alpha","beta","omega"],[get_A_indexes, get_alpha_indexes,get_beta_indexes,get_omega_indexes]):
            start_index,end_index = f(wave_index=wave_index,num_leads=num_leads,num_waves=num_waves)
            start_index_single_lead,_ = f(wave_index=wave_index,num_leads=1,num_waves=num_waves)
            # Some coefficients are the same for each lead
            if(coeff_name=="A" or coeff_name=="beta"): # For coefficients that have a different value for each lead
                single_lead_coeff_index = start_index+lead# Add the offset to get the coefficient for that lead
            if(coeff_name=="alpha" or coeff_name=="omega"): # For coefficients that have the same value for each lead
                single_lead_coeff_index = start_index# The index is always start_index
            extracted_lead_fmm_coefficients_array[start_index_single_lead] = fmm_coefficients_array[single_lead_coeff_index]
    start_index,end_index = get_M_indexes(wave_index=wave_index,num_leads=num_leads,num_waves=num_waves)
    start_index_single_lead,_ = get_M_indexes(wave_index=wave_index,num_leads=1,num_waves=num_waves)
    single_lead_coeff_index = start_index+lead # The M parameter has a different value for each lead
    extracted_lead_fmm_coefficients_array[start_index_single_lead] = fmm_coefficients_array[single_lead_coeff_index]
    return extracted_lead_fmm_coefficients_array

def convert_fmm_array_to_dict(fmm_array:np.ndarray,num_leads:int,num_waves:int)->Dict:
    # Quasi-Inverse of fmm_dict_to_array
    # Quasi inverse because alpha and omega parameters are scalars and not vectors (since they are the same for each lead)
    # However convert_fmm_dictionary_to_array assumed they were vectors with elements all equal to the same values
    parameters_dict = {}
    for i,w in enumerate(get_wave_names(num_waves=num_waves)):
        wave_dict = {}
        for coeff_name,f in zip(["A","alpha","beta","omega"],[get_A_indexes, get_alpha_indexes,get_beta_indexes,get_omega_indexes]):
            start_index,end_index = f(wave_index=i,num_leads=num_leads,num_waves=num_waves)
            wave_dict[coeff_name] = fmm_array[start_index:end_index]
        start_index,end_index = get_M_indexes(wave_index=None,num_leads=num_leads,num_waves=num_waves)
        wave_dict["M"] = fmm_array[start_index:end_index]
        parameters_dict[w] = wave_dict
    return parameters_dict

def convert_to_linear(x:float):
    # Equivalent to return x-np.pi*(np.sign(x-np.pi))
    if(x<np.pi and x>=0):
        x = x + np.pi
    elif(x>=np.pi and x<2*np.pi):
        x = x - np.pi
    elif(x<0 or x>2*np.pi):
        raise ValueError(f"Input should be between 0 and 2pi, found {x}")
    return x

def sort_fmm_coeffs_array(fmm_array:np.ndarray,num_leads,num_waves:int)->np.ndarray:
    # Sort fmm coefficient according to alpha coefficient
    # assert num_leads==1 #Only num_leads=1 and num_waves=5 are supported
    # assert num_waves==5
    num_samples = fmm_array.shape[0]
    all_samples_indexes = np.arange(num_samples)[:, None]
    alpha_indexes = np.array([get_alpha_indexes(x,num_leads,num_waves)[0] for x in range(5)])
    only_alpha_matrix = fmm_array[:,alpha_indexes]
    convert_to_linear_np = np.vectorize(convert_to_linear)
    only_alpha_matrix = convert_to_linear_np(only_alpha_matrix)
    order_alpha = np.argsort(only_alpha_matrix)
    rearranged_fmm_array = np.zeros_like(fmm_array)
    #TODO: we need to consider the lead if we want to extend the function to multiple leads
    for coeff_name,f in zip(["A","alpha","beta","omega"],[get_A_indexes, get_alpha_indexes,get_beta_indexes,get_omega_indexes]):  
        # Since for one lead all the parameters are scalar, just use the start index ([0])
        coeff_indexes = []
        for i in range(num_waves):
            coeff_indexes_start, coeff_indexes_end = f(wave_index=i,num_leads=num_leads,num_waves=num_waves)
            wave_coeffs_indexes = np.arange(coeff_indexes_start,coeff_indexes_end)
            coeff_indexes.append(wave_coeffs_indexes)
        coeff_indexes = np.array(coeff_indexes)
        ordered_coeff_indexes = coeff_indexes[order_alpha]
        reshaped_ordered_coeff_indexes = np.reshape(ordered_coeff_indexes,(fmm_array.shape[0],-1))
        reshaped_coeff_indexes = np.repeat(np.reshape(coeff_indexes,(1,-1)),repeats=num_samples,axis=0)
        rearranged_fmm_array[all_samples_indexes, reshaped_coeff_indexes] = fmm_array[all_samples_indexes, reshaped_ordered_coeff_indexes]

    # No need to modify M since it does not depends on the order
    m_start_index,m_end_index = get_M_indexes(wave_index=None,num_leads=num_leads,num_waves=num_waves)
    rearranged_fmm_array[:,m_start_index:m_end_index] = fmm_array[:,m_start_index:m_end_index]
    return rearranged_fmm_array

def get_loss_from_fmm_model(model,dataset,cfg, is_circular=False):
    num_batches = len(dataset)
    batch_size = cfg.batch_size
    waves = np.zeros((num_batches,batch_size,cfg.dataset.sequence_length,cfg.dataset.num_features,5))
    # m_parameter = np.zeros((num_batches,cfg.batch_size,cfg.dataset.num_features))
    num_pars_fun = get_fmm_num_parameters_circular if is_circular else get_fmm_num_parameters
    num_total_coefficient_per_sample,_ = num_pars_fun(num_leads=cfg.dataset.num_features,num_waves=cfg.model.num_waves)               
    loss_matrix = np.zeros((num_batches*cfg.batch_size,num_total_coefficient_per_sample))
    weighted_loss_matrix = np.zeros_like(loss_matrix)
    for batch_index,data in enumerate(dataset):
        loss_dict = model.compute_loss(model(data))
        loss_matrix[batch_index*batch_size:(batch_index+1)*batch_size,:] = loss_dict["coefficient_loss_vector"]
        weighted_loss_matrix[batch_index*batch_size:(batch_index+1)*batch_size,:] = loss_dict["coefficient_loss_vector_weighted"]
    return {"loss_matrix":loss_matrix,"weighted_loss_matrix":weighted_loss_matrix}

def reconstruct_FMM_leads_from_FMM_array(fmm_coeff_array, seq_len, num_leads, num_waves):
    # Get reconstructed signals from FMM coefficients array
    fmm_dict = convert_fmm_array_to_dict(fmm_array=fmm_coeff_array,num_leads=num_leads,num_waves=num_waves) 
    leads = np.zeros((seq_len,num_leads))    
    for lead in range(num_leads):
        waves = np.zeros((seq_len,num_waves))
        for i,wave_name in enumerate(get_wave_names(num_waves=num_waves)):
            wave = np.squeeze(generate_wave(fmm_dict,wave_name=wave_name,lead=lead,seq_len=seq_len))
            waves[:,i]=wave
        leads[:,lead] = fmm_dict["P"]["M"][lead] + np.sum(waves,axis=1)
    return leads

def expand_fmm_scalar_coefficients(fmm_dict, num_features):
    expanded_fmm_dict = {}
    for wave,wave_dict in fmm_dict.items():
        expanded_fmm_dict[wave] = {}
        for coeff_name, coeff in wave_dict.items():
            if(coeff_name in ["alpha","omega"]):
                expanded_fmm_dict[wave][coeff_name] = np.squeeze(np.repeat(coeff.numpy(), num_features))
            else:
                expanded_fmm_dict[wave][coeff_name] = np.squeeze(coeff.numpy())
    return expanded_fmm_dict

if __name__=='__main__':
    pass