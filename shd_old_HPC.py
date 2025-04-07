import numpy as np
import matplotlib.pyplot as plt
import mnist

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense,FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, LeakyIntegrateFire, SpikeInput
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

from ml_genn.compilers.event_prop_compiler import default_params

import logging

#logging.basicConfig(level=logging.DEBUG)

p= {
    "NUM_HIDDEN": 256,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 2,
    "REG_LAMBDA": 1e-10,
    "DT": 1.0,
    "KERNEL_PROFILING": True,
    "RECORDING": True,
    "NAME": "test",
    "OUT_DIR": ".",
    "SEED": 345
}

fname= f"{sys.argv[1]}.json"
with open(fname,"r") as f:
    p0= json.load(f)

for (name,value) in p0.items():
    p[name]= value
    
print(p)

np.random.seed(p["SEED"])

# Get SHD dataset
dataset = SHD(save_to='../data', train=True)
dataset_test = SHD(save_to='../data', train=False)

# Preprocess
spikes = []
labels = []
for i in range(len(dataset)):
    events, label = dataset[i]
    spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                          dataset.sensor_size))
    labels.append(label)

# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)

serialiser = Numpy(f"{p['OUT_DIR']}/{p['NAME']}_checkpoints")
network = Network(default_params)
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=p["BATCH_SIZE"] * max_spikes),
                       num_input, record_spikes=True)
    hidden = Population(LeakyIntegrateFire(v_thresh=1.0, tau_mem=20.0),
                        p["NUM_HIDDEN"], record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output, record_spikes=True)

    # Connections
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    Conn_Pop1_Pop1 = Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02)),
               Exponential(5.0))
    Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(latest_spike_time / p["DT"]))

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             reg_lambda_upper=p["REG_LAMBDA"], reg_lambda_lower=p["REG_LAMBDA"], 
                             reg_nu_upper=14, max_spikes=1500, 
                             optimiser=Adam(0.001), batch_size=p["BATCH_SIZE"], 
                             kernel_profiling=p["KERNEL_PROFILING"])
compiled_net = compiler.compile(network,f"{p['OUT_DIR']}/{p['NAME']}")

with compiled_net:
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    if p["RECORDING"]:
        callbacks = [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True),
            Checkpoint(serialiser),
        ]
    else:
        callbacks = [Checkpoint(serialiser)]

    for e in range(p["NUM_EPOCHS"]):
        metrics, cb_data  = compiled_net.train({input: spikes},
                                               {output: labels},
                                               num_epochs=1, shuffle=True,
                                               callbacks=callbacks)
        n0 = np.asarray(cb_data['spikes_hidden'])
        mean_n0 = np.mean(n0, axis = 0)
        std_n0 = np.std(n0, axis = 0)
        resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "a")
        resfile.write(f"{e} {np.count_nonzero(mean_n0==0)} {np.mean(mean_n0)} {np.mean(std_n0)} {metrics[output].result}\n")
        hidden_sg = compiled_net.connection_populations[Conn_Pop0_Pop1]
        hidden_sg.vars["g"].pull_from_device()
        g_view = hidden_sg.vars["g"].view.reshape((num_input, p["NUM_HIDDEN"]))
        g_view[:,hidden_spikes==0] += 0.002
        hidden_sg.vars["g"].push_to_device()
    compiled_net.save_connectivity((p["NUM_EPOCHS"] - 1,), serialiser)

    end_time = perf_counter()
    print(f"Accuracy = {100 * metrics[output].result}%")
    timefile = open( os.path.join(p["OUT_DIR"], p["NAME"]+"_timing.txt"), "a")
    if p["KERNEL_PROFILING"]:
        timefile.write("# Total time Neuron_update_time Presynaptic_update_time Gradient_batch_reduce_time Gradient_learn_time Reset_time Softmax1_time Softmax2_time Softmax3_time\n") 
        timefile.write(f"{compiled_net.genn_model.neuron_update_time} ")
        timefile.write(f"{compiled_net.genn_model.presynaptic_update_time} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('GradientLearn')} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('Reset')} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('BatchSoftmax1')} ")
        timefile.write(f"Softmax2 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax2')} ")
        timefile.write(f"Softmax3 time = {compiled_net.genn_model.get_custom_update_time('BatchSoftmax3')}\n")
    else:
        timefile.write(f"Time\n")
        timefile.write(f"{end_time - start_time}\n")

# Preprocess
spikes = []
labels = []
for i in range(len(dataset_test)):
    events, label = dataset_test[i]
    spikes.append(preprocess_tonic_spikes(events, dataset_test.ordering,
                                          dataset_test.sensor_size))
    labels.append(label)
    
# Determine max spikes and latest spike time
max_spikes = calc_max_spikes(spikes)
latest_spike_time = calc_latest_spike_time(spikes)
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Load network state from final checkpoint
network.load((0,), serialiser)
compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                             reset_in_syn_between_batches=True,
                             batch_size=p["BATCH_SIZE"])
compiled_net = compiler.compile(network,f"{p['OUT_DIR']}/{p['NAME']}")

with compiled_net:
    # Evaluate model on dataset_test
    start_time = perf_counter()
    metrics, _  = compiled_net.evaluate({input: spikes},
                                        {output: labels},
                                        callbacks=[])
    end_time = perf_counter()
    resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_test_results.txt"), "a")
    resfile.write(f"{metrics[output].result}\n")
    print(f"Accuracy = {100 * metrics[output].result}%")
    if not p["KERNEL_PROFILING"]:
        timefile.write(f"Time = {end_time - start_time}\n")
timefile.close()
