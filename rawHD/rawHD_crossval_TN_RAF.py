#export CUDA_PATH=/usr/local/cuda
import numpy as np
import csv
from tqdm import trange
import os
import pickle
import json
from datetime import datetime
import pandas as pd
import copy
import matplotlib.pyplot as plt
import re

from ml_genn import Network, Population, Connection
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, Callback, OptimiserParamSchedule
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, UserNeuron, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential
import augmentation_tools_spike_times as aug
from time import perf_counter
import sys

from ml_genn.utils.data import (calc_max_spikes,
                                preprocess_tonic_spikes)

import nvsmi
import json

from rawHD_dataset_loader_padded_spikes import rawHD_Loader

DEBUG = False
PLOTN = 10

with open(sys.argv[1], "r") as f: 
    params = json.load(f)
    
params["num_samples"] = None

if params["hetTau_mem"]:
    the_taum = np.random.uniform(params["tau_mem_min"],params["tau_mem_max"],params["NUM_HIDDEN"])
    raf_b = -1./the_taum
else:
    the_taum = params["tau_mem"]

if params["hetPeriod"]:
    the_p = np.random.uniform(params["period_min"],params["period_max"],params["NUM_HIDDEN"])
    
b_raf = -1. / the_taum
w_raf = 2. * np.pi / the_p

print(params["dataset_directory"])
x_train, y_train, z_train, x_test, y_test, z_test, x_validation, y_validation, z_validation = rawHD_Loader(dir = params["dataset_directory"],
                                                                                                           num_samples=params["num_samples"],
                                                                                                           shuffle = False,
                                                                                                           shuffle_seed = 0,
                                                                                                           process_padded_spikes = False,
                                                                                                           validation_split = 0.0)
training_details = pd.read_csv(os.path.expanduser(params.get("dataset_directory")) + "training_details.csv")
testing_details = pd.read_csv(os.path.expanduser(params.get("dataset_directory")) + "testing_details.csv")

schedule_epoch_total = 0

raf_neuron = UserNeuron(
        vars={
            "x": ("b*(-Isyn + x) - w * y","x"),
            "y": ("w * x + b * y", "0.0")},
        threshold="y - a_thresh",
        output_var_name="y",
        param_vals={"b": b_raf,
                    "w": w_raf,
                    "a_thresh": 1.0,
        },
        var_vals={"x": 0.0, "y": 0.0},
        sub_steps=100,
        solver="linear_euler"
    )

os.chdir("output")

# change dir for readout files
# lazy fix until a solution can be implemented with ml_genn to support output file directory change
try:
    os.makedirs(params.get("output_dir") + params.get("sweeping_suffix"))
except:
    pass

os.chdir(params.get("output_dir") + params.get("sweeping_suffix"))

# Create sequential model
network = Network()

# Determine max spikes and latest spike time (on validation data with margin - bit hacky)
max_spikes = 0
for events in x_train:
    max_spikes = max(max_spikes, len(x_train))
latest_spike_time = 1600
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

with network:
    # Populations
    input = Population(SpikeInput(max_spikes = params["BATCH_SIZE"] * max_spikes),
                       params["NUM_INPUT"],
                       record_spikes=True)
    
    hidden = Population(raf_neuron,
                        params.get("NUM_HIDDEN"), 
                        record_spikes=True)
    
    output = Population(LeakyIntegrate(tau_mem=params["tau_mem"], 
                                readout="avg_var_exp_weight"),
                params.get("NUM_OUTPUT"), 
                record_spikes=True)

    # Connections
    i2h = Connection(input, hidden, Dense(Normal(mean = params.get("input_hidden_w_mean"), 
                                            sd = params.get("input_hidden_w_sd"))),
                Exponential(params["tau_syn"]))
    
    if params.get("recurrent"):
        h2h = Connection(hidden, hidden, Dense(Normal(mean = params.get("hidden_hidden_w_mean"), 
                                                sd = params.get("hidden_hidden_w_sd"))),
                Exponential(params["tau_syn"]))
    
    h2o = Connection(hidden, output, Dense(Normal(mean = params.get("hidden_output_w_mean"),
                                sd = params.get("hidden_output_w_sd"))),
                Exponential(params["tau_syn"]))
    
clamp_weight_conns_dir = {i2h: (-10, 10), h2o: (-10, 10)}
if params["recurrent"] : clamp_weight_conns_dir = {i2h: (-10, 10), h2h: (-10, 10), h2o: (-10, 10)}

compiler = EventPropCompiler(example_timesteps = params.get("NUM_FRAMES") * params.get("INPUT_FRAME_TIMESTEP"),
                             losses="sparse_categorical_crossentropy",
                             batch_size = params.get("BATCH_SIZE"),
                             reg_lambda = params.get("reg_lambda"),
                             reg_nu_upper = params.get("reg_nu_upper"),
                             dt = params.get("dt"),
                             max_spikes=1500,
                             grad_limit= 50.0,)

#speaker_id = np.sort(training_details.Speaker.unique())
speaker_id = np.asarray([ 11, 2, 7, 3, 6, 8, 10, 0, 9, 1 ])
speaker_id = speaker_id.astype('int8')  #np where is fussy with int

# Create sequential model
speaker = list(training_details.loc[:, "Speaker"])

resfile= open("run_results.txt", "a")
resfile.write(f"# speaker_left epoch ")
resfile.write(f"hidden_n_zero mean_hidden_mean_spike std_hidden_mean_spike mean_hidden_std_spikes std_hidden_std_spikes ")
resfile.write(f"val_hidden_n_zero val_mean_hidden_mean_spike val_std_hidden_mean_spike val_mean_hidden_std_spikes ")
resfile.write(f"val_std_hidden_std_spikes ")
resfile.write(f"train_accuracy validation_accuracy\n")
resfile.close()

# Evaluate model on numpy dataset
start_time = perf_counter() 

for count, speaker_left in enumerate(speaker_id):
    compiled_net = compiler.compile(network, optimisers={"all_connections": {"weight": Adam(params.get("lr"))}},)
    
    print(f"speaker left: {speaker_left} ")

    fold_train = []
    fold_train_labels = []
    fold_val_spikes = []
    fold_val_labels = []
    for i in np.where(speaker != speaker_left)[0]:
        fold_train.append(x_train[i])
        fold_train_labels.append(y_train[i])
    for i in np.where(speaker == speaker_left)[0]:
        fold_val_spikes.append(preprocess_tonic_spikes(x_train[i], 
                                                       x_train[0].dtype.names,
                                                       (params["NUM_INPUT"], 1, 1),
                                                       time_scale = 1))        
        fold_val_labels.append(y_train[i])

    with compiled_net:
        # save parameters for reference
        json_object = json.dumps(params, indent = 4)
        with open("params.json", "w") as outfile:
            outfile.write(json_object)
        
        # main dictionaries for tracking data # TODO: fix so debugging can be switched off
        metrics, metrics_val, cb_data_training, cb_data_validation = {}, {}, {}, {}
        
        # alpha decay after 1st epoch at a rate of lr_decay_rate
        def alpha_schedule(epoch, alpha):
            if params["lr_decay"]:
                return params["lr"] * params["lr_decay_rate"]**epoch
            else:
                return alpha
    
        callbacks = []
        val_callbacks = []
        if params.get("recurrent"):
            callbacks.append(OptimiserParamSchedule("alpha", alpha_schedule))

        if params.get("verbose"):
            callbacks.append("batch_progress_bar")

        if params["record_spikeN"]:
            callbacks.append(SpikeRecorder(hidden,"shid", record_counts=True))
            val_callbacks.append(SpikeRecorder(hidden,"shid", record_counts=True))

        if DEBUG:
            callbacks.append(SpikeRecorder(input, key="spikes_input"))
            callbacks.append(VarRecorder(hidden, "x", key="x_hidden", neuron_filter=np.arange(PLOTN*PLOTN), example_filter=0))
            callbacks.append(VarRecorder(hidden, "y", key="y_hidden", neuron_filter=np.arange(PLOTN*PLOTN), example_filter=0))
            
            
        for e in range(params["NUM_EPOCH"]):
            e_train = copy.deepcopy(fold_train)
            e_train, e_train_labels = aug.merge_and_return_a_new(e_train, fold_train_labels, percentage_added = 1.0)
            e_train, e_train_labels = aug.augmentation_y_shift(e_train, e_train_labels, params["aug_shift"])
            e_train_spikes = []
            for events in e_train:
                e_train_spikes.append(preprocess_tonic_spikes(events, 
                                                              x_train[0].dtype.names,
                                                              (params["NUM_INPUT"], 1, 1),
                                                              time_scale = 1))
            metrics, val_metrics, cb_data, val_cb_data = compiled_net.train({input: e_train_spikes},
                                                                            {output: e_train_labels},
                                                                            start_epoch = e,
                                                                            num_epochs = 1,
                                                                            shuffle = True,
                                                                            callbacks = callbacks,
                                                                            validation_callbacks = val_callbacks,
                                                                            validation_x = {input: fold_val_spikes},
                                                                            validation_y = {output: fold_val_labels})  
            
            n0 = np.asarray(cb_data[f"shid"])
            n0_val = np.asarray(val_cb_data[f"shid"])
            mean_n0 = np.mean(n0, axis = 0)
            std_n0 = np.std(n0, axis = 0)
            mean_n0_val = np.mean(n0_val, axis = 0)
            std_n0_val = np.std(n0_val, axis = 0)                
            resfile= open("run_results.txt", "a")
            resfile.write(f"{speaker_left} {e} ")
            resfile.write(f"{np.count_nonzero(mean_n0==0)} {np.mean(mean_n0)} {np.std(mean_n0)} {np.mean(std_n0)} {np.std(std_n0)} ")
            resfile.write(f"{np.count_nonzero(mean_n0_val==0)} {np.mean(mean_n0_val)} {np.std(mean_n0_val)} {np.mean(std_n0_val)} ")
            resfile.write(f"{np.std(std_n0_val)} ")
            resfile.write(f"{metrics[output].result} {val_metrics[output].result}\n")
            if DEBUG:
                xBATCH = 0
                plt.figure()
                t_spk = np.asarray(cb_data["spikes_input"][0][xBATCH])
                id_spk = np.asarray(cb_data["spikes_input"][1][xBATCH])
                plt.scatter(t_spk, id_spk, s=2, marker="|")
                plt.title(f"input batch={0} epoch={e}")
                plt.figure()
                fig,ax = plt.subplots(PLOTN,PLOTN,sharex=True,sharey=True)
                for i in range(PLOTN):
                    for j in range(PLOTN):
                        id = i*PLOTN+j
                        x = np.asarray(cb_data["x_hidden"])[0,:,id].flatten()
                        y = np.asarray(cb_data["y_hidden"])[0,:,id].flatten()
                        ax[i][j].plot(x)
                        ax[i][j].plot(y)
                        plt.legend([f"x hid",f"y hid"])
                plt.show()
    end_time = perf_counter()
    print(f"Time = {end_time - start_time}s")
    
