import numpy as np
import copy
import matplotlib.pyplot as plt


from ml_genn import Connection, Network, Population
from ml_genn.callbacks import Checkpoint, SpikeRecorder, VarRecorder, Callback
from ml_genn.compilers import EventPropCompiler, InferenceCompiler
from ml_genn.connectivity import Dense,FixedProbability
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, UserNeuron, SpikeInput
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from ml_genn.readouts import AvgVarExpWeight
from time import perf_counter
from ml_genn.utils.data import (calc_latest_spike_time, calc_max_spikes,
                                preprocess_tonic_spikes)
from tonic.datasets import SMNIST
import sys
import os
import json
import logging
import mnist
import pickle

TRAIN = True
DEBUG_VARS = True
DEBUG_SPIKES = True
PLOTN = 10
#logging.basicConfig(level=logging.DEBUG)

p= {
    "NUM_INPUT": 79,
    "NUM_HIDDEN": 512, # ALIF neurons
    "TAU_MEM": 20.0,
    "TAU_A_MEAN": 700,
    "TAU_A_STD": 200,
    "DELTA_A": 0.2,
    "A_INI_RANGE": 3.0,
    "DT": 1.0,
    "LABEL": list(range(10)), #[4,9], 
    "NUM_EXAMPLES": 60000,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 300,
    "GRAD_LIMIT": 1000.0,
    "REG_LAMBDA_UPPER": 0.0,
    "REG_LAMBDA_LOWER": 0.0,
    "REG_NU_UPPER": 15,
    "DT": 1.0,
    "KERNEL_PROFILING": False,
    "OUT_DIR": ".",
    "SEED": 345,
    "TRIGGER_W": 1.0,#0.25, 
    "IN_HID_MEAN": 0.04,
    "IN_HID_STD": 0.02,
    "HID_HID_MEAN": 0.0,
    "HID_HID_STD": 0.01,
    "HID_OUT_MEAN": 0.0,
    "HID_OUT_STD": 0.1,
    "TAU_SYN": 5.0,
    "LR": 0.0001,
    "EX_FILTER": [ 32, 64, 96, 5032, 5064, 5096 ]
}

if len(sys.argv) == 2:
    p["NAME"] = sys.argv[1]
else:
    raise Exception(f"usage: {sys.argv[0]} <base name>")


dataset = SMNIST(save_to="../data", train=TRAIN, duplicate=False, num_neurons=p["NUM_INPUT"])

mnist.datasets_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
images = mnist.train_images()

spikes = []
labels = []
max_spikes = 0.0
latest_spike_time = 0.0
for i in range(len(dataset)):
    events, label = dataset[i]
    if label in p["LABEL"]:
        #print(images[i])
        #plt.figure()
        #plt.imshow(images[i])
        #print(events.shape)
        #print(events.dtype)
        extra = []
        for i in range (2*28):
            extra.append((785000+i*1000, p["NUM_INPUT"], 1))
        extra= np.asarray(extra, dtype=events.dtype)
        events = np.append(events,extra,axis=0)
        #plt.figure()
        #plt.scatter(events["t"],events["x"],s=2)
        #plt.show()
        # Calculate max spikes and max times
        max_spikes = max(max_spikes, len(events))
        latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)
        spikes.append(preprocess_tonic_spikes(events, dataset.ordering,
                                          dataset.sensor_size))
        labels.append(label)

num_examples = len(spikes)
num_examples = min(p["NUM_EXAMPLES"], num_examples)
max_example_timesteps = int((latest_spike_time+ p["TAU_MEM"])/ p["DT"])
print(f"max_example_timesteps: {max_example_timesteps}")

# DEBUG_MODE is meant to run only shortly and do some diagnostic plots
if DEBUG_VARS:
    p["NUM_EPOCHS"] = 6
    
try:
    fname= f"{sys.argv[1]}.json"
    with open(fname,"r") as f:
        p0= json.load(f)

    for (name,value) in p0.items():
        p[name]= value
except:
    print("No json file found, proceeding with standard settings")


#np.random.seed(p["SEED"])

print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_output = 10

serialiser = Numpy(f"{p['OUT_DIR']}/{p['NAME']}_checkpoints")

w_inhid = np.random.normal(p["IN_HID_MEAN"], p["IN_HID_STD"],(p["NUM_INPUT"]+1,p["NUM_HIDDEN"]))
w_inhid[p["NUM_INPUT"],:] = p["TRIGGER_W"]


alif_neurons = UserNeuron(
    vars={
        "v": ("(Isyn - v)/tau_mem", "0.0"),
        "A": ("(-A)/tau_A", "A + dA")},
    threshold="v - thresh - A",
    output_var_name="v",
    param_vals={"tau_mem": p["TAU_MEM"],
                "tau_A": np.random.normal(p["TAU_A_MEAN"],p["TAU_A_STD"],p["NUM_HIDDEN"]),
                "thresh": 1.0,
                "dA": p["DELTA_A"]
    },
    var_vals={"v": 0.0, "A": 0.0}, #np.random.uniform(0.0, p["A_INI_RANGE"],p["NUM_HIDDEN"])},
    solver="exponential_euler"
)

lif_neurons = UserNeuron(vars={"v": ("(Isyn - v)/tau_mem", "0.0")},
                        threshold="v - thresh",
                        output_var_name="v",
                        param_vals={"tau_mem": p["TAU_MEM"],"thresh": 1.0,},
                        var_vals={"v": 0})

network = Network()

with open(f"{p['NAME']}_run.json","w") as f:
    json.dump(p,f,indent=4)

with network:
    # Populations
    input = Population(SpikeInput(max_spikes=p["BATCH_SIZE"] * max_spikes),
                       p["NUM_INPUT"]+1, record_spikes=DEBUG_SPIKES)
    hidden = Population(alif_neurons, p["NUM_HIDDEN"], record_spikes=True)

    window_start = (max_example_timesteps-56)*p["DT"]-p["TAU_MEM"]
    window_end = max_example_timesteps*p["DT"]
    ro = AvgVarExpWeight(window_start=window_start, window_end=window_end)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout=ro),
                        num_output)

    # Connections
    inhid = Connection(input, hidden, Dense(w_inhid),
                        Exponential(p["TAU_SYN"]))
    hidhid = Connection(hidden, hidden, Dense(Normal(mean=p["HID_HID_MEAN"],
                                                         sd=p["HID_HID_STD"])),
                          Exponential(p["TAU_SYN"]))
    hidout = Connection(hidden, output, Dense(Normal(mean=p["HID_OUT_MEAN"],
                                                       sd=p["HID_OUT_STD"])),
                         Exponential(p["TAU_SYN"]))

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             reg_lambda=(p["REG_LAMBDA_UPPER"],p["REG_LAMBDA_LOWER"]),
                             grad_limit=p["GRAD_LIMIT"],
                             reg_nu_upper= p["REG_NU_UPPER"], max_spikes=1500, 
                             batch_size=p["BATCH_SIZE"], 
                             kernel_profiling=p["KERNEL_PROFILING"])

timefile = open( os.path.join(p["OUT_DIR"], p["NAME"]+"_timing.txt"), "w")
if p["KERNEL_PROFILING"]:
    timefile.write("# Total_time Neuron_update_time Presynaptic_update_time Gradient_batch_reduce_time ")
    timefile.write("Gradient_learn_time Reset_time Softmax1_time Softmax2_time Softmax3_time\n") 
else:
    timefile.write(f"# Total_time\n")
resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "w")
resfile.write(f"# Epoch ")
resfile.write(f"hidden_n_zero mean_hidden_mean_spike ")
resfile.write(f"val_hidden_n_zero val_mean_hidden_mean_spike ")
resfile.write(f"train_accuracy validation_accuracy\n")
resfile.close()

optimisers= {"all_connections": {"weight": Adam(p["LR"])},}
compiled_net = compiler.compile(network,f"{p['OUT_DIR']}/{p['NAME']}",
                                optimisers=optimisers)
with compiled_net:
    # Evaluate model
    start_time = perf_counter()
    if DEBUG_SPIKES:
        callbacks = [
            SpikeRecorder(input, key="spikes_input"),
            SpikeRecorder(hidden, key="spikes_hidden"),
        ]
        val_callbacks = [
            SpikeRecorder(hidden, key="spikes_hidden"),
        ]
    else:
        callbacks = [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True),
        ]
        val_callbacks = [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True),
        ]
        
    if DEBUG_VARS:
        callbacks.append(VarRecorder(hidden, "v", key=f"v_hidden", neuron_filter=np.arange(PLOTN*PLOTN), example_filter= p["EX_FILTER"]))
        callbacks.append(VarRecorder(hidden, "Lambdav", key=f"lambdav_hidden", neuron_filter=np.arange(PLOTN*PLOTN), example_filter= p["EX_FILTER"]))
        callbacks.append(VarRecorder(hidden, "A", key=f"A_hidden", neuron_filter=np.arange(PLOTN*PLOTN), example_filter= p["EX_FILTER"]))
        callbacks.append(VarRecorder(hidden, "LambdaA", key=f"lambdaA_hidden", neuron_filter=np.arange(PLOTN*PLOTN), example_filter= p["EX_FILTER"]))
        #callbacks.append(VarRecorder(hidden, "Lambdai", key=f"lambdaI_hidden", neuron_filter=np.arange(PLOTN*PLOTN), example_filter= p["EX_FILTER"]))
        callbacks.append(VarRecorder(hidden, "out_post", key=f"i_hidden", neuron_filter=np.arange(PLOTN*PLOTN), example_filter= p["EX_FILTER"]))
     
        callbacks.append(VarRecorder(output, "v", key="v_output", example_filter= p["EX_FILTER"]))
        callbacks.append(VarRecorder(output, "Lambdav", key="lambdav_output", example_filter= p["EX_FILTER"]))
        
    for e in range(p["NUM_EPOCHS"]):
        print(f"Training {num_examples} examples")
        metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train_validate({input: spikes[:num_examples]},
                                                                         {output: labels[:num_examples]},
                                                                         num_epochs=1, start_epoch=e,shuffle=False,
                                                                         callbacks=callbacks,validation_split= 0.1,
                                                                         validation_callbacks=val_callbacks)
        
        
        if DEBUG_SPIKES:
            #print(cb_data['spikes_hidden'][0])
            #exit(1)
            n0 = np.asarray([[np.sum(cb_data[f"spikes_hidden"][1][i] == j) for j in range(p[f"NUM_HIDDEN"])] for i in range(len(cb_data[f"spikes_hidden"][1]))])
            n0_val = np.asarray([[np.sum(val_cb_data[f"spikes_hidden"][1][i] == j) for j in range(p[f"NUM_HIDDEN"])] for i in range(len(val_cb_data[f"spikes_hidden"][1]))])
        else:
            n0 = np.asarray(cb_data[f"spikes_hidden"])
            n0_val = np.asarray(val_cb_data[f"spikes_hidden"])
        resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "a")
        resfile.write(f"{e} ")
        mean_n0 = np.mean(n0, axis = 0)
        #plt.figure()
        #plt.hist(mean_n0, bins = p["NUM_HIDDEN"])
        #plt.show()
        mean_n0_val = np.mean(n0_val, axis = 0)
        resfile.write(f"{np.count_nonzero(mean_n0==0)} {np.mean(mean_n0)} ")
        resfile.write(f"{np.count_nonzero(mean_n0_val==0)} {np.mean(mean_n0_val)} ")
        resfile.write(f"{metrics[output].result} {val_metrics[output].result}\n")
        resfile.close()
        for ex,n in enumerate(p["EX_FILTER"]):
            if DEBUG_SPIKES or DEBUG_VARS:
                plt.figure()
                plt.imshow(images[n])
                plt.title(f"input ex {n}, epoch={e}")
            if DEBUG_SPIKES:
                t_spk = np.asarray(cb_data['spikes_input'][0][n])
                id_spk = np.asarray(cb_data['spikes_input'][1][n])
                plt.figure()
                plt.scatter(t_spk, id_spk, s=2, marker="|")
                plt.xlim([ 0, max_example_timesteps*p["DT"]])
                plt.title(f"input ex {n}, epoch={e}")
                t_spk = np.asarray(cb_data[f"spikes_hidden"][0][n])
                id_spk = np.asarray(cb_data[f"spikes_hidden"][1][n])
                plt.figure()
                plt.scatter(t_spk, id_spk, s=2, marker="|")
                plt.xlim([ 0, max_example_timesteps*p["DT"]])
                plt.title(f"hidden ex {n}, epoch={e}")
            if DEBUG_VARS:
                fig, ax = plt.subplots(PLOTN,PLOTN,sharex=True,sharey=True)
                for i in range(PLOTN):
                    for j in range(PLOTN):
                        ID = i*PLOTN+j
                        #ax[i,j].set_xlim([0, p["DT"]*784+56+20.0])
                        v = np.asarray(cb_data[f"v_hidden"])[ex,:,ID].flatten()
                        ax[i,j].plot(v)
                        A = np.asarray(cb_data[f"A_hidden"])[ex,:,ID].flatten()
                        ax[i,j].plot(A)
                        I = np.asarray(cb_data[f"i_hidden"])[ex,:,ID].flatten()
                        ax[i,j].plot(I)
                plt.suptitle(f"hid fwd {n}")
                fig, ax = plt.subplots(PLOTN,PLOTN,sharex=True,sharey=True)
                for i in range(PLOTN):
                    for j in range(PLOTN):
                        ID = i*PLOTN+j
                        #ax[i,j].set_xlim([0, p["DT"]*784+56+20.0])
                        v = np.asarray(cb_data[f"lambdav_hidden"])[ex,:,ID].flatten()
                        ax[i,j].plot(v)
                        A = np.asarray(cb_data[f"lambdaA_hidden"])[ex,:,ID].flatten()
                        ax[i,j].plot(A)
                plt.suptitle(f"hid bwd {n-p['BATCH_SIZE']}")
                fig, ax = plt.subplots(5,2,sharex=True,sharey=True)
                for i in range(5):
                    for j in range(2):
                        ID = i*2+j
                        #ax[i,j].set_xlim([0, p["DT"]*784+56+20.0])
                        v = np.asarray(cb_data[f"v_output"])[ex,:,ID].flatten()
                        ax[i,j].plot(v)
                plt.suptitle(f"out fwd {n}")
                fig, ax = plt.subplots(5,2)
                for i in range(5):
                    for j in range(2):
                        ID = i*2+j
                        #ax[i,j].set_xlim([0, p["DT"]*784+56+20.0])
                        v = np.asarray(cb_data[f"lambdav_output"])[ex,:,ID].flatten()
                        ax[i,j].plot(v)
                plt.suptitle(f"out bwd {n-p['BATCH_SIZE']}")
        if DEBUG_SPIKES or DEBUG_VARS:
            plt.show()
    end_time = perf_counter()
    timefile = open( os.path.join(p["OUT_DIR"], p["NAME"]+"_timing.txt"), "a")
    if p["KERNEL_PROFILING"]:
        timefile.write(f"{end_time - start_time} ")        
        timefile.write(f"{compiled_net.genn_model.neuron_update_time} ")
        timefile.write(f"{compiled_net.genn_model.presynaptic_update_time} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('GradientBatchReduce')} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('GradientLearn')} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('Reset')} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('BatchSoftmax1')} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('BatchSoftmax2')} ")
        timefile.write(f"{compiled_net.genn_model.get_custom_update_time('BatchSoftmax3')}\n")
    else:
        timefile.write(f"{end_time - start_time}\n")
    timefile.close()
    if DEBUG_MODE:
        plt.show()
