import numpy as np
import matplotlib.pyplot as plt
import mnist

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
import copy

#logging.basicConfig(level=logging.DEBUG)

p= {
    "NUM_HIDDEN": 256,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 100,
    "REG_LAMBDA": 2e-5,
    "GRAD_LIMIT": 100.0,
    "REG_NU_UPPER": 14,
    "DT": 1.0,
    "KERNEL_PROFILING": True,
    "NAME": "test",
    "OUT_DIR": ".",
    "SEED": 345,
    "N_DELAY": 10,
    "T_DELAY": 30,
    "AUGMENT_SHIFT": 40,
    "P_BLEND": 0.5,
    "N_BLEND": 2000,
}

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

class Blend:
    def __init__(self, p_blend, n_blend, sensor_size):
        self.p_blend = p_blend
        self.n_blend = n_blend
        self.sensor_size = sensor_size
    def __call__(self, dataset: list, classes: list) -> list:
        ds = copy.deepcopy(dataset)
        for i in range(self.n_blend):
            idx = np.random.randint(0,len(ds))
            idx2 = np.random.randint(0,len(classes[ds[idx][1]]))
            assert ds[idx][1] == ds[classes[ds[idx][1]][idx2]][1]
            ds.append((self.blend(ds[idx][0], ds[classes[ds[idx][1]][idx2]][0]), ds[idx][1]))

        return ds

    def blend(self, X1, X2):
        X1 = copy.deepcopy(X1)
        X2 = copy.deepcopy(X2)
        mx1 = np.mean(X1["x"])
        mx2 = np.mean(X2["x"])
        mt1 = np.mean(X1["t"])
        mt2 = np.mean(X2["t"])
        X1["x"]+= int((mx2-mx1)/2)
        X2["x"]+= int((mx1-mx2)/2)
        X1["t"]+= int((mt2-mt1)/2)
        X2["t"]+= int((mt1-mt2)/2)
        X1 = np.delete(
            X1,
            np.where(
                (X1["x"] < 0) | (X1["x"] >= self.sensor_size[0]) | (X1["t"] < 0) | (X1["t"] >= 1000000)))
        X2 = np.delete(
            X2,
            np.where(
                (X2["x"] < 0) | (X2["x"] >= self.sensor_size[0]) | (X2["t"] < 0) | (X2["t"] >= 1000000)))
        mask1 = np.random.rand(X1["x"].shape[0]) < self.p_blend
        mask2 = np.random.rand(X2["x"].shape[0]) < self.p_blend
        X1_X2 = np.concatenate([X1[mask1], X2[mask2]])
        idx= np.argsort(X1_X2["t"])
        X1_X2 = X1_X2[idx]
        return X1_X2

class Delay:
    def __init__(self, t_delay, n_delay, num_input):
        self.n_delay = n_delay
        self.t_delay = t_delay * 1000
        self.num_input = num_input
    def __call__(self, events: np.ndarray) -> np.ndarray:
        delayed_events = [copy.deepcopy(events)]

        for n in range(1, self.n_delay):
            curr_event = copy.deepcopy(events)
            curr_event["x"] = curr_event["x"] + (n * self.num_input)
            curr_event["t"] = curr_event["t"] + (n * self.t_delay)
            curr_event = np.delete(curr_event, np.where(curr_event["t"] >= 1000000))
            delayed_events.append(curr_event)
        return np.concatenate(delayed_events)

if len(sys.argv) > 1:
    fname= f"{sys.argv[1]}.json"
    with open(fname,"r") as f:
        p0= json.load(f)

    for (name,value) in p0.items():
        p[name]= value
    
print(p)

np.random.seed(p["SEED"])

# Get SHD dataset
dataset = SHD(save_to='../data', train=True)
# Get number of input and output neurons from dataset 
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)
speakers = dataset.speaker
spklist = np.unique(speakers)

delay = Delay(p["T_DELAY"], p["N_DELAY"], num_input)
shift = Shift(p["AUGMENT_SHIFT"], dataset.sensor_size)
blend = Blend(p["P_BLEND"], p["N_BLEND"], dataset.sensor_size)

serialiser = Numpy(f"{p['OUT_DIR']}/{p['NAME']}_checkpoints")

qif_neuron = UserNeuron(vars={"v": ("(v*(v-v_c) + Isyn) / tau_mem", "0.0")},
                        threshold="v - 1.0",
                        output_var_name="v",
                        param_vals={"tau_mem": 20.0, "v_c": 0.5},
                        var_vals={"v": 0.0})

# calculate an estimate for max_spikes in input neurons
max_spikes = 0
latest_spike_time = 0
for events, label in dataset:
    events = np.delete(events, np.where(events["t"] >= 1000000))
    d_events = delay(events)
    max_spikes = max(max_spikes, len(d_events))
    latest_spike_time = max(latest_spike_time, np.amax(d_events["t"]) / 1000.0)

# add a generous margin as blending could lead to more spikes 
max_spikes = int(max_spikes*1.2)
print(f"Overall max_spikes limit: {max_spikes}")

network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=p["BATCH_SIZE"] * max_spikes),
                       num_input*p["N_DELAY"], record_spikes=True)
    hidden = Population(qif_neuron,
                        p["NUM_HIDDEN"], record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output, record_spikes=True)

    # Connections
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=0.001, sd=0.0003)),
               Exponential(5.0))
    #Conn_Pop1_Pop1 = Connection(hidden, hidden, Dense(Normal(mean=0.0, sd=0.02)),
    #           Exponential(5.0))
    Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
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
    train = [dataset[i] for i in np.where(speakers != left)[0]]
    valid = [dataset[i] for i in np.where(speakers == left)[0]]
    # determine classes content for blending
    classes = [[] for _ in range(20)]
    max_spikes = 0
    latest_spike_time = 0
    for i in range(len(train)):
        events, label = train[i]
        events = np.delete(events, np.where(events["t"] >= 1000000))
        classes[label].append(i)
        d_events = delay(events)
        # Calculate max spikes and max times
        max_spikes = max(max_spikes, len(d_events))
        latest_spike_time = max(latest_spike_time, np.amax(d_events["t"]) / 1000.0)

    spikes_val = []
    labels_val = []
    for i in range(len(valid)):
        events, label = valid[i]
        events = np.delete(events, np.where(events["t"] >= 1000000))
        spikes_val.append(preprocess_tonic_spikes(delay(events), dataset.ordering,
                                                  (dataset.sensor_size[0]*p["N_DELAY"],
                                                   dataset.sensor_size[1],
                                                   dataset.sensor_size[2])))
        labels_val.append(label)
    max_spikes = max(max_spikes, calc_max_spikes(spikes_val))
    latest_spike_time = max(latest_spike_time, calc_latest_spike_time(spikes_val))    
    print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

    compiled_net = compiler.compile(network,f"{p['OUT_DIR']}/{p['NAME']}")
    with compiled_net:
        # Evaluate model on numpy dataset
        start_time = perf_counter()
        callbacks = [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True),
            #VarRecorder(hidden,"SpikeCountBackBatch",key="scnt"),
            Checkpoint(serialiser), EaseInSchedule()
        ]
        val_callbacks =  [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True)
        ]

        for e in range(p["NUM_EPOCHS"]):
            # Apply augmentation to events and preprocess
            spikes_train = []
            labels_train = []
            blended_dataset = blend(train, classes)
            for events, label in blended_dataset:
            #for events, label in train:
                spikes_train.append(preprocess_tonic_spikes(delay(shift(events)), dataset.ordering,
                                                            (dataset.sensor_size[0]*p["N_DELAY"],
                                                             dataset.sensor_size[1],dataset.sensor_size[2])))
                labels_train.append(label)
            if e == 0:
                print(f"Leave speaker {left}: training with {len(spikes_train)} examples and validating with {len(spikes_val)}.")

            metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train({input: spikes_train},
                                                                             {output: labels_train},
                                                                             validation_x={input: spikes_val},
                                                                             validation_y={output: labels_val},
                                                                             num_epochs=1, shuffle=True,
                                                                             callbacks=callbacks,
                                                                             validation_callbacks=val_callbacks)
            n0 = np.asarray(cb_data['spikes_hidden'])
            #for k in range(5):
            #    fig, ax = plt.subplots(10,10,sharex=True, sharey=True)
            #    for i in range(10):
            #        for j in range(10):
            #            ax[i,j].hist(n0[(k*10+i)*10+j])
            #            print((k*10+i)*10+j)
            #scnt = np.asarray(cb_data['scnt'])
            #print(scnt.shape)
            #if e < 2:
            #    plt.figure()
            #    plt.plot(scnt[:,0,:])
            #    plt.show()
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
            g_view = hidden_sg.vars["weight"].view.reshape((num_input*p["N_DELAY"], p["NUM_HIDDEN"]))
            g_view[:,mean_n0==0] += 0.0002
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
