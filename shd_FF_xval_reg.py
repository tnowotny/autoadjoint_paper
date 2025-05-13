import numpy as np
#import matplotlib.pyplot as plt
#import mnist

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, Callback
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

class EaseInSchedule(Callback):
    def set_params(self, compiled_network, **kwargs):
        self._optimisers = [o for o, _ in compiled_network.optimisers]

    def on_batch_begin(self, batch):
        # Set parameter to return value of function
        for optimiser in self._optimisers:
            if optimiser.alpha < 0.001 :
                optimiser.alpha = (0.001 / 1000.0) * (1.05 ** batch)
            else:
                optimiser.alpha = 0.001

#logging.basicConfig(level=logging.DEBUG)

p= {
    "NUM_HIDDEN": 256,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 100,
    "GRAD_LIMIT": 100.0,
    "REG_LAMBDA": 5e-6,
    "REG_NU_UPPER": 14,
    "DT": 1.0,
    "KERNEL_PROFILING": False,
    "NAME": "FFWD_alif_maass_4",
    "OUT_DIR": ".",
    "SEED": 345
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

lif_neuron = UserNeuron(vars={"v": ("Isyn + a - b * v", "c")},
                        threshold="v - v_thr",
                        output_var_name="v",
                        param_vals={"a": 0, "b": 1/20, "c": 0, "v_thr": 1},
                        var_vals={"v": 0})

# initial weight values for LIF:
init_w_lif = {"in_hid": (0.0015, 0.0005),
              "hid_out": (0.0, 0.03)}

# BALAZS: is the reset of g to e correct? Not g+e?
alif_tn_neuron = UserNeuron(vars={"v": ("Isyn + a - b * v + g * (d - v)", "c"), "g":("-g / tau", "e")},
                        threshold="v - v_thr",
                        output_var_name="v",
                        param_vals={"a": 0, "b": 1/20, "c": 0, "d": 0, "e": 0.2, "tau": 200, "v_thr": 1},
                        var_vals={"v": 0, "g": 0})

# Thomas' version: g is reset to g+e 
alif_neuron = UserNeuron(vars={"v": ("Isyn + a - b * v + g * (d - v)", "c"), "g":("-g / tau", "g + e")},
                        threshold="v - v_thr",
                        output_var_name="v",
                        param_vals={"a": 0, "b": 1/20, "c": 0, "d": 0, "e": 0.2, "tau": 200, "v_thr": 1},
                        var_vals={"v": 0, "g": 0})

# alif neuron from Maass, "Spike frequency adaptation supports network computations on temporally dispersed information", eLife
# taum = 20
# vth on the order of 10-30 
# tau_a = 1200, beta= 3; tau_a = 2000, beta = 1
alif_maass_neuron = UserNeuron(vars={"v": ("Isyn + a - b * v", "c"), "g":("-g / tau", "g+e")},
                        threshold="v - v_thr - g",
                        output_var_name="v",
                        param_vals={"a": 0, "b": 1/20, "c": 0, "d": 0, "e": 0.1, "tau": 200, "v_thr": 1},
                        var_vals={"v": 0, "g": 0})



# raf doesn'work currently (maybe just bad parameter choices)
raf_neuron = UserNeuron(vars={"x": ("Isyn + b * x - w * y", "0"), "y": ("w * x + b * y", "1")},
                        threshold="y - a_thresh",
                        output_var_name="x",
                        param_vals={"b":-1, "w": 10, "a_thresh":1},
                        var_vals={"x":0, "y":0})

qif_neuron = UserNeuron(vars={"v": ("(v*(v-v_c) + Isyn) / tau_mem", "0.0")},
                        threshold="v - 1.0",
                        output_var_name="v",
                        param_vals={"tau_mem": 20.0, "v_c": 0.5},
                        var_vals={"v": 0.0})

# initial weight values for QIF:
init_w_qif = {"in_hid": (0.03, 0.01),
              "hid_out": (0.0, 0.03)}

network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=p["BATCH_SIZE"] * max_spikes),
                       num_input, record_spikes=True)
    hidden = Population(alif_neuron,
                        p["NUM_HIDDEN"], record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output, record_spikes=True)

    # Connections
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=init_w_lif["in_hid"][0], sd=init_w_lif["in_hid"][1])),
               Exponential(5.0))
    Connection(hidden, output, Dense(Normal(mean=init_w_lif["hid_out"][0], sd=init_w_lif["hid_out"][1])),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(latest_spike_time / p["DT"]))

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             reg_lambda=p["REG_LAMBDA"],
                             grad_limit=p["GRAD_LIMIT"],
                             reg_nu_upper= p["REG_NU_UPPER"], max_spikes=1500, 
                             optimiser=Adam(0.001*0.001), batch_size=p["BATCH_SIZE"], 
                             kernel_profiling=p["KERNEL_PROFILING"],
                             solver="linear_euler")

timefile = open( os.path.join(p["OUT_DIR"], p["NAME"]+"_timing.txt"), "w")
if p["KERNEL_PROFILING"]:
    timefile.write("# Speaker_left Total_time Neuron_update_time Presynaptic_update_time Gradient_batch_reduce_time Gradient_learn_time Reset_time Softmax1_time Softmax2_time Softmax3_time\n") 
else:
    timefile.write(f"# Total_time\n")
resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "w")
resfile.write(f"# Speaker_left Epoch hidden_n_zero mean_hidden_mean_spike std_hidden_mean_spike mean_hidden_std_spikes std_hidden_std_spikes val_hidden_n_zero val_mean_hidden_mean_spike val_std_hidden_mean_spike val_mean_hidden_std_spikes val_std_hidden_std_spikes train_accuracy validation_accuracy\n")
resfile.close()

for left in spklist:
    train_spikes = [spikes[i] for i in np.where(speakers != left)[0]]
    train_labels = [labels[i] for i in np.where(speakers != left)[0]]
    val_spikes = [spikes[i] for i in np.where(speakers == left)[0]]
    val_labels = [labels[i] for i in np.where(speakers == left)[0]]
    print(f"Leave speaker {left}: training with {len(train_spikes)} examples and validating with {len(val_spikes)}.") 
    compiled_net = compiler.compile(network,f"{p['OUT_DIR']}/{p['NAME']}")
    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True),
            Checkpoint(serialiser), EaseInSchedule(),
        ]
        val_callbacks =  [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True)
        ]
        early_stop, best_acc = 15, 0
        for e in range(p["NUM_EPOCHS"]):
            metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train({input: train_spikes},
                                                                             {output: train_labels},
                                                                             validation_x={input: val_spikes},
                                                                             validation_y={output: val_labels},
                                                                             num_epochs=1, shuffle=True,
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
            if metrics[output].result > best_acc:
                best_acc = metrics[output].result
                early_stop = 15
            else:
                early_stop -= 1
            if early_stop == 0:
                break
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
