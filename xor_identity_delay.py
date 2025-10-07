import numpy as np
#import matplotlib.pyplot as plt
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
from make_diagnostic_data import generate_xor_data_identity_coding, generate_xor_data_identity_coding_Poisson

from time import perf_counter
from ml_genn.utils.data import preprocess_spikes
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

#logging.basicConfig(level=logging.DEBUG)

p= {
    "NUM_HIDDEN": 1024,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 300,
    "GRAD_LIMIT": 100.0,
    "REG_LAMBDA": 1e-6,
    "REG_NU_UPPER": 0.0,
    "DT": 1.0,
    "KERNEL_PROFILING": False,
    "NAME": "",
    "OUT_DIR": ".",
    "SEED": 345,
    "HIDDEN_NEURONS": "raf_thomas",
    "TAU_A_MIN": 25,
    "TAU_A_MAX": 500,
    "T_DIGIT": 20.0,
    "IN_DELAY": 90.0,
    "N_TRAIN": 1000,
    "N_VAL": 100,
    "N_TEST": 100,
    "R_NOISE": 0.001,
    "MIN_W_RAF": 0.07,
    "MAX_W_RAF": 0.08,
    "MIN_B_RAF": -0.05,
    "MAX_B_RAF": -0.01,
    "W_LIFT": 0.0002,
    "R_LOW": 0.01,
    "R_HIGH": 0.2,
    "W_IH_RAF_THOMAS": (0.0, 0.12),
    "W_HO_RAF_THOMAS": (0.0, 0.03),
    "DATA_METHOD": "Sample",
    "N_SPIKE": 5,
    "T_REFR": 2.0,
    "W_IH_LIF": (0.12, 0.04),
    "W_HO_LIF": (0.0, 0.03)
}

t_total = 3*p["T_DIGIT"]+p["IN_DELAY"]

if len(sys.argv) != 2:
    raise Exception(f"usage: {sys.argv[0]} <base name>")

try:
    fname= f"{sys.argv[1]}.json"
    with open(fname,"r") as f:
        p0= json.load(f)        
    for (name,value) in p0.items():
        p[name]= value
except:
    print("No valid json file found, proceeding with standard settings")

# override naming from JSON file if in conflict with command line
p["NAME"] = sys.argv[1]

DEBUG_MODE = False
# DEBUG_MODE is meant to run only shortly and do some diagnostic plots
if DEBUG_MODE:
    p["N_TRAIN"] = 128
    p["N_VAL"] = 128
    p["N_TEST"] = 32
    p["NUM_EPOCHS"] = 2
    RECORDING = True
    xBATCH = 0
else:
    RECORDING = False
    

print(p)
with open(f"{p['NAME']}_run.json","w") as f:
    json.dump(p,f,indent=4)


# load data
# generate data
if p["DATA_METHOD"] == "Sample":   
    t_train, id_train, lab_train = generate_xor_data_identity_coding(t_total,p["T_DIGIT"], p["N_SPIKE"], p["T_REFR"], p["R_LOW"], p["N_TRAIN"])
    t_val, id_val, lab_val = generate_xor_data_identity_coding(t_total,p["T_DIGIT"], p["N_SPIKE"], p["T_REFR"], p["R_LOW"], p["N_VAL"])
    t_test, id_test, lab_test = generate_xor_data_identity_coding(t_total,p["T_DIGIT"], p["N_SPIKE"], p["T_REFR"], p["R_LOW"], p["N_TEST"])
elif p["DATA_METHOD"] == "Poisson":
    t_tain, id_train, lab_train = generate_xor_data_identity_coding_Poisson(t_total,p["T_DIGIT"], p["R_LOW"], p["R_HIGH"], p["N_TRAIN"])
    t_val, id_val, lab_val = generate_xor_data_identity_coding_Poisson(t_total,p["T_DIGIT"], p["R_LOW"], p["R_HIGH"], p["N_VAL"])
    t_test, id_test, lab_test = generate_xor_data_identity_coding_Poisson(t_total,p["T_DIGIT"], p["R_LOW"], p["R_HIGH"], p["N_TEST"])
else:
    print(f"Unkown data method {p['DATA_METHOD']}")
    exit(1)
    
# Determine max spikes and latest spike time
# calculate an estimate for max_spikes in input neurons
max_spikes = 0
for d in [t_train, t_val, t_test]:
    for st in d:
        max_spikes = max(max_spikes, len(st))

# add a safety margin to max_spikes as we will be regenerating training data
max_spikes = int(1.5*max_spikes)

# also round the latest spike time for the same reason
        
print(f"Max spikes {max_spikes}, latest spike time {t_total}")

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = 4
num_output = 2

serialiser = Numpy(f"{p['OUT_DIR']}/{p['NAME']}_checkpoints")


w_bohte = np.random.uniform(p["MIN_W_RAF"],p["MAX_W_RAF"],p["NUM_HIDDEN"])
b_bohte = np.random.uniform(p["MIN_B_RAF"],p["MAX_B_RAF"],p["NUM_HIDDEN"])
#b_bohte = 100*(-1+np.sqrt(1-(0.1*w_bohte)**2))/0.1
if DEBUG_MODE:
    print(w_bohte)
    print(b_bohte)
    plt.figure()
    plt.scatter(w_bohte, b_bohte)
    plt.show()

neurons= {
"if": UserNeuron(vars={"v": ("Isyn", "c")},
                 threshold="v - v_thr",
                 output_var_name="v",
                 param_vals={"c": 0, "v_thr": 1},
                 var_vals={"v": 0}),
"lif": UserNeuron(vars={"v": ("Isyn + a - b * v", "c")},
                  threshold="v - v_thr",
                  output_var_name="v",
                  param_vals={"a": 0, "b": 1/20, "c": 0, "v_thr": 1},
                  var_vals={"v": 0}),
# alif neuron with somewhat unusual setting g to e upon spikes
"alif_balazs": UserNeuron(vars={"v": ("Isyn + a - b * v + g * (d - v)", "c"), "g":("-g / tau", "e")},
                          threshold="v - v_thr",
                          output_var_name="v",
                          param_vals={"a": 0, "b": 1/20, "c": 0, "d": 0, "e": 0.2, "tau": np.random.uniform(low=p["TAU_A_MIN"],high=p["TAU_A_MAX"],size=p["NUM_HIDDEN"]), "v_thr": 1},
                          var_vals={"v": 0, "g": 0}),
# alif with the more common jump of adding g + e
"alif_thomas":  UserNeuron(vars={"v": ("Isyn + a - b * v + b * g * (d - v)", "c"), "g":("-g / tau", "g + e")},
                         threshold="v - v_thr",
                         output_var_name="v",
                         param_vals={"a": 0, "b": 1/20, "c": 0, "d": 0, "e": 0.2, "tau": np.random.uniform(low=p["TAU_A_MIN"],high=p["TAU_A_MAX"],size=p["NUM_HIDDEN"]), "v_thr": 1},
                         var_vals={"v": 0, "g": 0}),
# alif neuron from Maass, "Spike frequency adaptation supports network computations on temporally dispersed information", eLife
# taum = 20
# vth on the order of 10-30 
# tau_a = 1200, beta= 3; tau_a = 2000, beta = 1
"alif_maass": UserNeuron(vars={"v": ("Isyn + a - b * v", "c"), "g":("-g / tau", "g+e")},
                         threshold="v - v_thr - g",
                         output_var_name="v",
                         param_vals={"a": 0, "b": 1/20, "c": 0, "d": 0, "e": 0.1, "tau": np.random.uniform(low=p["TAU_A_MIN"],high=p["TAU_A_MAX"],size=p["NUM_HIDDEN"]), "v_thr": 1},
                         var_vals={"v": 0, "g": 0}),
# raf doesn'work currently (maybe just bad parameter choices)
"raf": UserNeuron(vars={"x": ("Isyn + b * x - w * y", "0"), "y": ("w * x + b * y", "1")},
                  threshold="y - a_thresh",
                  output_var_name="x",
                  param_vals={"b":-0.1, "w": 0.1, "a_thresh":1},
                  var_vals={"x": 0, "y": 0},
                  sub_steps=100),
# raf according to Bohte. Soft reset and refractory period
"raf_bohte": UserNeuron(vars={"x": ("Isyn + (b-q) * x - w * y", "x"), "y": ("w * x + (b-q) * y", "y"), "q": ("-gma*q", "q+1")},
                  threshold="y - (a_thresh+q)",
                  output_var_name="x",
                  param_vals={"b": b_bohte, "w": w_bohte, "a_thresh": 1, "gma": -np.log(0.8)},
                  var_vals={"x": 0, "y": 0},
                  sub_steps=100),
"raf_thomas": UserNeuron(vars={"x": ("Isyn + b * x - w * y", "x"), "y": ("w * x + b * y", "y"), "q": ("-gma*q", "q+1")},
                  threshold="y - (a_thresh+q)",
                  output_var_name="y",
                  param_vals={"b": b_bohte, "w": w_bohte, "a_thresh": 1, "gma": -np.log(0.8)},
                  var_vals={"x": 0.0, "y": 0.0, "q": 0.0},
                  sub_steps=100),
"qif": UserNeuron(vars={"v": ("(v*(v-v_c) + Isyn) / tau_mem", "0.0")},
                  threshold="v - 1.0",
                  output_var_name="v",
                  param_vals={"tau_mem": 20.0, "v_c": 0.5},
                  var_vals={"v": 0.0}),
}

init_vals = {
    "if": {"in_hid": (0.0015, 0.0005),
            "hid_out": (0.0, 0.03)},
    "lif": {"in_hid": p["W_IH_LIF"],
            "hid_out": p["W_HO_LIF"]},
    "alif_balazs": {"in_hid": (0.0015, 0.0005),
                  "hid_out": (0.0, 0.03)},
    "alif_thomas": {"in_hid": (0.0015, 0.0005),
                  "hid_out": (0.0, 0.03)},
    "alif_maass": {"in_hid": (0.0015, 0.0005),
                  "hid_out": (0.0, 0.03)},
    "raf": {"in_hid": (0.03, 0.01),
                  "hid_out": (0.0, 0.03)},
    "raf_bohte": {"in_hid": (0.0, 0.004),
                  "hid_out": (0.0, 0.03)},
    "raf_thomas": {"in_hid": p["W_IH_RAF_THOMAS"],
                   "hid_out": p["W_HO_RAF_THOMAS"]},
    "qif": {"in_hid": (0.03, 0.01),
            "hid_out": (0.0, 0.03)},
}

network = Network()

hn = p["HIDDEN_NEURONS"]
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=p["BATCH_SIZE"] * max_spikes),
                       num_input, record_spikes=RECORDING)
    hidden = Population(neurons[hn],
                        p["NUM_HIDDEN"], record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output)

    # Connections
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=init_vals[hn]["in_hid"][0],
                                                            sd=init_vals[hn]["in_hid"][1])),
                                Exponential(5.0))
    Conn_Pop1_Pop2 = Connection(hidden, output, Dense(Normal(mean=init_vals[hn]["hid_out"][0],
                                            sd=init_vals[hn]["hid_out"][1])),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(t_total / p["DT"]))

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             reg_lambda=p["REG_LAMBDA"],
                             grad_limit=p["GRAD_LIMIT"],
                             reg_nu_upper= p["REG_NU_UPPER"], max_spikes=1500, 
                             optimiser=Adam(0.001*0.001), batch_size=p["BATCH_SIZE"], 
                             kernel_profiling=p["KERNEL_PROFILING"])

timefile = open( os.path.join(p["OUT_DIR"], p["NAME"]+"_timing.txt"), "w")
if p["KERNEL_PROFILING"]:
    timefile.write("# Total_time Neuron_update_time Presynaptic_update_time Gradient_batch_reduce_time Gradient_learn_time Reset_time Softmax1_time Softmax2_time Softmax3_time\n") 
else:
    timefile.write(f"# Total_time\n")
resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "w")
resfile.write(f"# Epoch hidden_n_zero mean_hidden_mean_spike std_hidden_mean_spike mean_hidden_std_spikes std_hidden_std_spikes val_hidden_n_zero val_mean_hidden_mean_spike val_std_hidden_mean_spike val_mean_hidden_std_spikes val_std_hidden_std_spikes train_accuracy validation_accuracy\n")
resfile.close()

compiled_net = compiler.compile(network,f"{p['OUT_DIR']}/{p['NAME']}",optimisers={Conn_Pop0_Pop1: {"weight": Adam(0.001*0.001)},
                                                                                  Conn_Pop1_Pop2: {"weight": Adam(0.001*0.001)},
                                                                                  hidden: {"b": Adam(0.001*0.001),
                                                                                           "w": Adam(0.001*0.001)}})
                                                                                  
                                                                                  
with compiled_net:
    # Evaluate model on numpy dataset
    start_time = perf_counter()
    if DEBUG_MODE:
        callbacks = [
            SpikeRecorder(input, key="spikes_input", example_filter=[ 0, 1]),
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True),
            VarRecorder(hidden, "x", key="x_hidden", neuron_filter=[0,1], example_filter=0),
            VarRecorder(hidden, "y", key="y_hidden", neuron_filter=[0,1], example_filter=0),
            Checkpoint(serialiser), EaseInSchedule(),
        ]
        val_callbacks = [
            SpikeRecorder(input, key="spikes_input"),
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True),
        ]
    else:
        callbacks = [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True),
            Checkpoint(serialiser), EaseInSchedule(),
        ]

        val_callbacks =  [
            SpikeRecorder(hidden, key="spikes_hidden",record_counts=True)
        ]
    #early_stop, best_acc = 50, 0
    spikes_val = []
    for t, ids in zip(t_val, id_val):        
        spikes_val.append(preprocess_spikes(np.asarray(t), ids, num_input))
            
    for e in range(p["NUM_EPOCHS"]):
        # fresh training data each epoch to achieve good sampling
        if p["DATA_METHOD"] == "Sample":   
            t_train, id_train, lab_train = generate_xor_data_identity_coding(t_total,p["T_DIGIT"], p["N_SPIKE"], p["T_REFR"], p["R_LOW"], p["N_TRAIN"])
        elif p["DATA_METHOD"] == "Poisson":
            t_train, id_train, lab_train = generate_xor_data_identity_coding_Poisson(t_total,p["T_DIGIT"], p["R_LOW"], p["R_HIGH"], p["N_TRAIN"])
        spikes_train = []
        for t, ids in zip(t_train, id_train):
            spikes_train.append(preprocess_spikes(np.asarray(t), ids, num_input))
        print(f"Training {len(spikes_train)} examples and validating with {len(spikes_val)}")
        metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train({input: spikes_train},
                                                                         {output: lab_train},
                                                                         validation_x={input: spikes_val},
                                                                         validation_y={output: lab_val},
                                                                         num_epochs=1, start_epoch=e, shuffle=True,
                                                                         callbacks=callbacks,
                                                                         validation_callbacks=val_callbacks)
        n0 = np.asarray(cb_data['spikes_hidden'])
        n0_val = np.asarray(val_cb_data['spikes_hidden'])
        mean_n0 = np.mean(n0, axis = 0)
        std_n0 = np.std(n0, axis = 0)
        mean_n0_val = np.mean(n0_val, axis = 0)
        std_n0_val = np.std(n0_val, axis = 0)                
        resfile= open(os.path.join(p["OUT_DIR"], p["NAME"]+"_results.txt"), "a")
        resfile.write(f"{e} {np.count_nonzero(mean_n0==0)} {np.mean(mean_n0)} {np.std(mean_n0)} {np.mean(std_n0)} {np.std(std_n0)} {np.count_nonzero(mean_n0_val==0)} {np.mean(mean_n0_val)} {np.std(mean_n0_val)} {np.mean(std_n0_val)} {np.std(std_n0_val)} {metrics[output].result} {val_metrics[output].result}\n")
        resfile.close()
        #if metrics[output].result > best_acc:
        #    best_acc = metrics[output].result
        #    early_stop = 50
        #else:
        #    early_stop -= 1
        #if early_stop == 0:
        #    break
        hidden_sg = compiled_net.connection_populations[Conn_Pop0_Pop1]
        hidden_sg.vars["weight"].pull_from_device()
        g_view = hidden_sg.vars["weight"].view.reshape((num_input, p["NUM_HIDDEN"]))
        g_view[:,mean_n0==0] += p["W_LIFT"]
        hidden_sg.vars["weight"].push_to_device()
        if DEBUG_MODE:
            t_spk = np.asarray(cb_data['spikes_input'][0][xBATCH])
            id_spk = np.asarray(cb_data['spikes_input'][1][xBATCH])
            plt.figure()
            plt.scatter(t_spk, id_spk, s=1, marker="|")
            plt.title(f"input batch={xBATCH} epoch={e}")
            plt.figure()
            x = np.asarray(cb_data['x_hidden'])[0,:,0].flatten()
            y = np.asarray(cb_data['y_hidden'])[0,:,0].flatten()
            plt.plot(x)
            plt.plot(y)
            x = np.asarray(cb_data['x_hidden'])[0,:,1].flatten()
            y = np.asarray(cb_data['y_hidden'])[0,:,1].flatten()
            plt.plot(x)
            plt.plot(y)
            plt.legend(["x hidden","y hidden"]) 
            
    compiled_net.save_connectivity(serialiser)
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
