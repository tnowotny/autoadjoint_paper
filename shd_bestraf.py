import numpy as np
import copy

from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder, VarRecorder
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

from argparse import ArgumentParser

parser = ArgumentParser()
#these values correspond to the best model that I found
parser.add_argument("--max_w", type=float, default=0.1, help="Maximum omega")
parser.add_argument("--max_b", type=float, default=-1.0, help="Maximum b")
parser.add_argument("--min_w", type=float, default=0.0045, help="Minimum omega")
parser.add_argument("--min_b", type=float, default=-0.0007, help="Minimum b")
parser.add_argument("--k_reg", type=float, default=5e-12, help="Spike regularisation strength")
parser.add_argument("--grad_limit", type=int, default=100, help="Gradient clipping limit")
parser.add_argument("--weight_lr", type=float, default=0.001, help="Weight learning rate")
parser.add_argument("--w_lr", type=float, default=0.01, help="Omega learning rate")
parser.add_argument("--b_lr", type=float, default=0.01, help="b learning rate")
parser.add_argument("--solver", type=str, default="linear_euler", help="Solver type")
parser.add_argument("--substeps", type=int, default=100, help="Number of substeps for solver")
parser.add_argument("--speaker", type=int, default=0, help="Speaker to leave out for cross-validation (4 and 5 are only in the test set)")
args = parser.parse_args()

unique_suffix = "_".join(("_".join(str(i) for i in val) if isinstance(val, list) 
                         else str(val))
                         for arg, val in vars(args).items())
NUM_HIDDEN = 256
BATCH_SIZE = 32
NUM_EPOCHS = 300
DT = 1.0

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


w_raf = np.random.uniform(-0.5,0.5,NUM_HIDDEN)
b_raf = np.random.uniform(-0.5,0.5,NUM_HIDDEN)

# Get SHD dataset
dataset = SHD(save_to='../data', train=True)

# Preprocess
spikes_train = []
labels_train = []
spikes_val = []
labels_val = []
max_spikes = 0
latest_spike_time = 0.0
for i in range(len(dataset)):
    events, label = dataset[i]
    if dataset.speaker[i] != args.speaker:
        spikes_train.append(events)
        labels_train.append(label)
        max_spikes = max(max_spikes, len(events))
        latest_spike_time = max(latest_spike_time, np.amax(events["t"]) / 1000.0)
    else:
        spikes_val.append(preprocess_tonic_spikes(events, dataset.ordering,
                                                dataset.sensor_size))
        labels_val.append(label)

# Determine max spikes and latest spike time
max_spikes = max(max_spikes, calc_max_spikes(spikes_val))
latest_spike_time = max(latest_spike_time, calc_latest_spike_time(spikes_val))

# Get number of input and output neurons from dataset 
# and round up outputs to power-of-two
num_input = int(np.prod(dataset.sensor_size))
num_output = len(dataset.classes)

serialiser = Numpy("checkpoints_" + unique_suffix)
# make augmentations
shift = Shift(40, dataset.sensor_size)

# raf neurons
raf_neuron = UserNeuron(vars={"x": ("Isyn + ((max_b / (1 + e ** (-b))) + min_b) * x - ((max_w / (1 + e ** (-w))) + min_w) * y", "x"), "y": ("((max_w / (1 + e ** (-w))) + min_w) * x + ((max_b / (1 + e ** (-b))) - min_b) * y", "y")},
                        threshold="y - a_thresh",
                        output_var_name="y",
                        param_vals={"b": b_raf, "w": w_raf, "a_thresh": 1, "max_b": args.max_b, "min_b": args.min_b, "max_w": args.max_w, "min_w": args.min_w, "e": np.exp(1)},
                        var_vals={"x": 0.0, "y": 0.0, "q": 0.0},
                        sub_steps=100,
                        solver="linear_euler"
)


network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input, record_spikes=True)
    hidden = Population(raf_neuron,
                        NUM_HIDDEN, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=20.0, readout="avg_var_exp_weight"),
                        num_output, record_spikes=True)

    # Connections
    Conn_Pop0_Pop1 = Connection(input, hidden, Dense(Normal(mean=0.03, sd=0.01)),
               Exponential(5.0))
    Connection(hidden, output, Dense(Normal(mean=0.0, sd=0.03)),
               Exponential(5.0))

max_example_timesteps = int(np.ceil(latest_spike_time / DT))

compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                             losses="sparse_categorical_crossentropy",
                             reg_lambda= args.k_reg,
                             grad_limit=args.grad_limit,
                             reg_nu_upper=1, max_spikes=1500, 
                             batch_size=BATCH_SIZE)
    
model_name = (f"classifier_train_{md5(unique_suffix.encode()).hexdigest()}"
                if os.name == "nt" else f"classifier_train_{unique_suffix}")
compiled_net = compiler.compile(network,model_name, optimisers= {"all_connections": {"weight": Adam(args.weight_lr)}, hidden: {"b": Adam(args.b_lr), "w": Adam(args.w_lr)}})

results_dic = {}
with compiled_net:
    callbacks = [SpikeRecorder(hidden, key="spikes_hidden",record_counts=True)]
    val_callbacks =  []
    warmup = True
    start_epoch = 0
    acc = None
    while warmup:
        start_epoch += 1
        spikes_train_shift = []
        labels_train_shift = []
        for i in range(len(spikes_train)):
            spikes_train_shift.append(preprocess_tonic_spikes(shift(spikes_train[i]), dataset.ordering, dataset.sensor_size))
            labels_train_shift.append(labels_train[i])
        metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train({input: spikes_train_shift},
                                                                            {output: labels_train_shift},
                                                                            validation_x={input: spikes_val},
                                                                            validation_y={output: labels_val},
                                                                            num_epochs=1, start_epoch=start_epoch-1, shuffle=True,
                                                                            callbacks=callbacks,
                                                                            validation_callbacks=val_callbacks)
        hidden_spikes = np.zeros(NUM_HIDDEN)
        for cb_d in cb_data['spikes_hidden']:
              hidden_spikes += cb_d


        hidden_sg = compiled_net.connection_populations[Conn_Pop0_Pop1]
        hidden_sg.vars["weight"].pull_from_device()
        g_view = hidden_sg.vars["weight"].view.reshape((num_input, NUM_HIDDEN))
        g_view[:,hidden_spikes==0] += 0.002
        hidden_sg.vars["weight"].push_to_device()   
        if acc is None:
            acc = val_metrics[output].result
        elif acc < val_metrics[output].result:
            acc = val_metrics[output].result
            warmup = False
            compiled_net.save(("best",), serialiser)
        results_dic["train_acc_warmup"] = str(metrics[output].result)
        results_dic["val_acc_warmup"] = str(val_metrics[output].result)
        results_dic["meanspike_warmup"] = str(np.mean(hidden_spikes))
        results_dic["nonspike_warmup"] = str(np.sum(hidden_spikes==0))
        results_dic["epoch_warmup"] = str(start_epoch)
        with open(f"results/acc_raf_{unique_suffix}.json", 'w') as f:
                json.dump(results_dic, f, indent=4)
        if start_epoch >= 50:
            quit()
    early_stop = 15
    for e in range(start_epoch,NUM_EPOCHS):
        spikes_train_shift = []
        labels_train_shift = []
        for i in range(len(spikes_train)):
            spikes_train_shift.append(preprocess_tonic_spikes(shift(spikes_train[i]), dataset.ordering, dataset.sensor_size))
            labels_train_shift.append(labels_train[i])
        metrics, val_metrics, cb_data, val_cb_data  = compiled_net.train({input: spikes_train_shift},
                                                                            {output: labels_train_shift},
                                                                            validation_x={input: spikes_val},
                                                                            validation_y={output: labels_val},
                                                                            num_epochs=1, start_epoch=e, shuffle=True,
                                                                            callbacks=callbacks,
                                                                            validation_callbacks=val_callbacks)
        
        
        hidden_spikes = np.zeros(NUM_HIDDEN)
        for cb_d in cb_data['spikes_hidden']:
              hidden_spikes += cb_d
        
        hidden_sg = compiled_net.connection_populations[Conn_Pop0_Pop1]
        hidden_sg.vars["weight"].pull_from_device()
        g_view = hidden_sg.vars["weight"].view.reshape((num_input, NUM_HIDDEN))
        g_view[:,hidden_spikes==0] += 0.002
        hidden_sg.vars["weight"].push_to_device()
        if val_metrics[output].result > acc:
            acc = val_metrics[output].result
            results_dic["train_acc"] = str(metrics[output].result)
            results_dic["val_acc"] = str(val_metrics[output].result)
            early_stop = 15
            compiled_net.save(("best",), serialiser)
            results_dic["meanspike"] = str(np.mean(hidden_spikes))
            results_dic["nonspike"] = str(np.sum(hidden_spikes==0))
            with open(f"results/acc_raf_{unique_suffix}.json", 'w') as f:
                json.dump(results_dic, f, indent=4)
        else:
            early_stop -= 1
            if early_stop < 0:
                break
        

