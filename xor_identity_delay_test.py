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
from make_diagnostic_data import generate_xor_data_identity_coding

from time import perf_counter
from ml_genn.utils.data import preprocess_spikes
import sys
import os
import json

import logging

RECORDING = True
p= {
    "NUM_HIDDEN": 128,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 600,
    "GRAD_LIMIT": 100.0,
    "REG_LAMBDA": 1e-9,
    "REG_NU_UPPER": 0,
    "DT": 1.0,
    "KERNEL_PROFILING": False,
    "NAME": "",
    "OUT_DIR": ".",
    "SEED": 345,
    "HIDDEN_NEURONS": "lif",
    "TAU_A_MIN": 25,
    "TAU_A_MAX": 500,
    "T_DIGIT": 20.0,
    "IN_DELAY": 0.0,
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
    "R_HIGH": 1.0
}


if len(sys.argv) == 2:
    p["NAME"] = sys.argv[1]
else:
    raise Exception(f"usage: {sys.argv[0]} <base name>")

try:
    fname= f"{sys.argv[1]}_run.json"
    with open(fname,"r") as f:
        p0= json.load(f)

    for (name,value) in p0.items():
        p[name]= value
except:
    print("No json file found, proceeding with standard settings")

print(p)
t_total = 3*p["T_DIGIT"]+p["IN_DELAY"]
print(f"t_total: {t_total}")

# load data
# generate data
t_test, id_test, lab_test = generate_xor_data_identity_coding(t_total,p["T_DIGIT"], p["R_LOW"], p["R_HIGH"], p["N_TEST"])

# Determine max spikes and latest spike time
# calculate an estimate for max_spikes in input neurons
max_spikes = 0
for st in t_test:
    max_spikes = max(max_spikes, len(st))


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
    "lif": {"in_hid": (0.24, 0.08),
            "hid_out": (0.0, 0.03)},
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
    "raf_thomas": {"in_hid": (0.0, 0.15),
                  "hid_out": (0.0, 0.03)},
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
    Connection(hidden, output, Dense(Normal(mean=init_vals[hn]["hid_out"][0],
                                            sd=init_vals[hn]["hid_out"][1])),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(t_total / p["DT"]))

network.load((p["NUM_EPOCHS"] - 1,), serialiser)

compiler = InferenceCompiler(evaluate_timesteps=max_example_timesteps,
                             reset_in_syn_between_batches=True,
                             batch_size=p["BATCH_SIZE"], dt=p["DT"])
compiled_net = compiler.compile(network)


with compiled_net:
    callbacks = [
        SpikeRecorder(input, key="spikes_input"),
    ]
    start_time = perf_counter()
    spikes_test = []
    for t, ids in zip(t_test, id_test):
        spikes_test.append(preprocess_spikes(np.asarray(t), ids, num_input))
    print(f"Testing {len(spikes_test)} examples.")
    pred, cb_data  = compiled_net.predict({input: spikes_test},
                                             output, callbacks=callbacks)
    #print(cb_data["spikes_input"][0])
    #print(pred)
    end_time = perf_counter()
    good = 0
    #fig = plt.figure()
    clr = ['r', 'b' ]
    pred = pred[output]
    for i in range(len(pred)):
        pr = np.argmax(pred[i])
        if pr == lab_test[i]:
            good += 1
        else:
            plt.figure()
            plt.scatter(cb_data["spikes_input"][0][i],cb_data["spikes_input"][1][i])
            plt.show()
        #fig.plot(pred[i,0],pred[i,1],'o',color=clr[lab_test[i]])
    acc = good/len(pred)
    print(f"accuracy: {acc}")
    plt.show()
