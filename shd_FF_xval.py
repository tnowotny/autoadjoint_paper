import numpy as np
#import matplotlib.pyplot as plt
import copy

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense,FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, UserNeuron, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from tonic.datasets import SHD

from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)
import sys
import os
import json

import logging

#logging.basicConfig(level=logging.DEBUG)

class Shift:
    def __init__(self, f_shift, sensor_size):
        self.f_shift = f_shift
        self.sensor_size = sensor_size

    def __call__(self, events: np.ndarray) -> np.ndarray:
        # Shift events
        events_copy = copy.deepcopy(events)
        events_copy["x"] = events_copy["x"] + np.random.randint(-self.f_shift, self.f_shift)

        # Delete out of bound events
        events_copy = np.delete(
            events_copy,
            np.where(
                (events_copy["x"] < 0) | (events_copy["x"] >= self.sensor_size[0])))
        return events_copy

p= {
    "NUM_HIDDEN": 256,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 100,
    "GRAD_LIMIT": 50.0,
    "DT": 1.0,
    "KERNEL_PROFILING": False,
    "NAME": "test9",
    "OUT_DIR": ".",
    "SEED": 345,
    "MIN_W_RAF": 0.07,
    "MAX_W_RAF": 0.08,
    "MIN_B_RAF": -0.05,
    "MAX_B_RAF": -0.01,
    "REG_LAMBDA": 5e-7,
    "IN_HID_MEAN": 0.015,
    "IN_HID_STD": 0.005,
    "ETA": 0.0005,
    "AUGMENT_SHIFT": 20.0
}

if len(sys.argv) > 1:
    fname= f"{sys.argv[1]}.json"
    with open(fname,"r") as f:
        p0= json.load(f)

    for (name,value) in p0.items():
        p[name]= value
    
print(p)
with open(f"{p['NAME']}_run.json","w") as f:
    json.dump(p,f,indent=4)

np.random.seed(p["SEED"])
w_bohte = np.random.uniform(p["MIN_W_RAF"],p["MAX_W_RAF"],p["NUM_HIDDEN"])
b_bohte = np.random.uniform(p["MIN_B_RAF"],p["MAX_B_RAF"],p["NUM_HIDDEN"])

# Get SHD dataset
dataset = SHD(save_to='../data', train=True)

# Preprocess
spikes = []
labels = []
for i in range(len(dataset)):
    events, label = dataset[i]
    spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                          dataset.sensor_size))
    labels.append(label)
speakers = dataset.speaker
spklist = np.unique(speakers)

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)

serialiser = Numpy(f"{p['OUT_DIR']}/{p['NAME']}_checkpoints")

# make augmentations
shift = Shift(p["AUGMENT_SHIFT"], dataset.sensor_size)

# raf neurons
raf_neuron = UserNeuron(vars={"x": ("Isyn + b * x - w * y", "x"), "y": ("w * x + b * y", "y"), "q": ("-gma*q", "q+1")},
                        threshold="y - (a_thresh+q)",
                        output_var_name="y",
                        param_vals={"b": b_bohte, "w": w_bohte, "a_thresh": 1, "gma": -np.log(0.8)},
                        var_vals={"x": 0.0, "y": 0.0, "q": 0.0},
                        sub_steps=100,
                        solver="linear_euler"
)

network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=p["BATCH_SIZE"] * max_spikes),
                       num_input, record_spikes=True)
    hidden = Population(raf_neuron,
                        p["NUM_HIDDEN"], record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output, record_spikes=True)

    # Connections
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=p["IN_HID_MEAN"], sd=p["IN_HID_STD"])),
               Exponential(5.0))
    Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(latest_spike_time / p["DT"]))

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             reg_lambda= p["REG_LAMBDA"],
                             grad_limit=p["GRAD_LIMIT"],
                             reg_nu_upper=14.0, max_spikes=1500, 
                             batch_size=p["BATCH_SIZE"], 
                             kernel_profiling=p["KERNEL_PROFILING"])

timefile = open( os.path.join(p["OUT_DIR"], p["NAME"]+"_timing.txt"), "w")
if p["KERNEL_PROFILING"]:
    timefile.write("# Speaker_left Total_time Neuron_update_time Presynaptic_update_time Gradient_batch_reduce_time Gradient_learn_time Reset_time Softmax1_time Softmax2_time Softmax3_time\n") 
else:
    timefile.write(f"# Total_time\n")
resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "w")
resfile.write(f"# Speaker_left Epoch hidden_n_zero mean_hidden_mean_spike std_hidden_mean_spike mean_hidden_std_spikes std_hidden_std_spikes val_hidden_n_zero val_mean_hidden_mean_spike val_std_hidden_mean_spike val_mean_hidden_std_spikes val_std_hidden_std_spikes train_accuracy validation_accuracy\n")
resfile.close()

for left in spklist:
    # Preprocess
    spikes_train = []
    labels_train = []
    spikes_val = []
    labels_val = []
    for i in range(len(dataset)):
        events, label = dataset[i]
        if speakers[i] != left:
            spikes_train.append(preprocess_tonic_spikes(shift(events), dataset.ordering, dataset.sensor_size))
            labels_train.append(label)
        else:
            spikes_val.append(preprocess_tonic_spikes(shift(events), dataset.ordering, dataset.sensor_size))
            labels_val.append(label)
        
    print(f"Leave speaker {left}: training with {len(spikes_train)} examples and validating with {len(spikes_val)}.") 
    compiled_net = compiler.compile(network,f"{p['OUT_DIR']}/{p['NAME']}", optimisers= {"all_connections": {"weight": Adam(p["ETA"])}, hidden: {"b": Adam(p["ETA"]), "w": Adam(p["ETA"])}})
    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True),
            #Checkpoint(serialiser),
        ]
        val_callbacks =  [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True)
        ]
        for e in range(p["NUM_EPOCHS"]):
            metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train({input: spikes_train},
                                                                             {output: labels_train},
                                                                             validation_x={input: spikes_val},
                                                                             validation_y={output: labels_val},
                                                                             num_epochs=1, start_epoch=e, shuffle=True,
                                                                             callbacks=callbacks,
                                                                             validation_callbacks=val_callbacks)
            n0 = np.asarray(cb_data['spikes_hidden'])
            mean_n0 = np.mean(n0, axis = 0)
            std_n0 = np.std(n0, axis = 0)
            n0_val = np.asarray(val_cb_data['spikes_hidden'])
            mean_n0_val = np.mean(n0_val, axis = 0)
            std_n0_val = np.std(n0_val, axis = 0)
            resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "a")
            resfile.write(f"{left} {e} {np.count_nonzero(mean_n0==0)} {np.mean(mean_n0)} {np.std(mean_n0)} {np.mean(std_n0)} {np.std(std_n0)} {np.count_nonzero(mean_n0_val==0)} {np.mean(mean_n0_val)} {np.std(mean_n0_val)} {np.mean(std_n0_val)} {np.std(std_n0_val)} {metrics[output].result} {val_metrics[output].result}\n")
            resfile.close()
            hidden_sg = compiled_net.connection_populations[Conn_Pop0_Pop1]
            hidden_sg.vars["weight"].pull_from_device()
            g_view = hidden_sg.vars["weight"].view.reshape((num_input, p["NUM_HIDDEN"]))
            g_view[:,mean_n0==0] += 0.002
            hidden_sg.vars["weight"].push_to_device()            
        compiled_net.save_connectivity((left,), serialiser)
        end_time = perf_counter()
        timefile = open( os.path.join(p["OUT_DIR"], p["NAME"]+"_timing.txt"), "a")
        if p["KERNEL_PROFILING"]:
            timefile.write(f"{left} {end_time - start_time} ")        
            timefile.write(f"{compiled_net.genn_model.neuron_update_time} ")
            timefile.write(f"{compiled_net.genn_model.presynaptic_update_time} ")
            timefile.write(f"{compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')} ")
            timefile.write(f"{compiled_net.genn_model.get_custom_update_time('GradientLearn')} ")
            timefile.write(f"{compiled_net.genn_model.get_custom_update_time('Reset')} ")
            timefile.write(f"{compiled_net.genn_model.get_custom_update_time('BatchSoftmax1')} ")
            timefile.write(f"{compiled_net.genn_model.get_custom_update_time('BatchSoftmax2')} ")
            timefile.write(f"{compiled_net.genn_model.get_custom_update_time('BatchSoftmax3')}\n")
        else:
            timefile.write(f"{left} {end_time - start_time}\n")
        timefile.close()
