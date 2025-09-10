import numpy as np
import matplotlib.pyplot as plt

from ml_genn import Connection, Network, Population
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import Dense
from ml_genn.initializers import Normal
from ml_genn.neurons import LeakyIntegrate, SpikeInput, LeakyIntegrateFire
from ml_genn.optimisers import Adam
from ml_genn.synapses import Exponential
from ml_genn.callbacks import SpikeRecorder, VarRecorder

from time import perf_counter
from ml_genn.utils.data import preprocess_spikes

import copy


BATCH_SIZE = 1
NUM_EPOCHS = 40
DT = 1.0
TRIAL_T = 100.0
TRAIN = True

spikes = []
ind = np.array([[0,1],[1,0],[0,1],[1,0]])
time = np.array([[0,10],[0,10],[0,10],[0,10]])
#ind = np.array([[0,1],[1,0],[0,1],[1,0]])
#time = np.array([[5,5],[5,5],[5,5],[5,5]])
labels = np.array([0,1,0,1])
for t, i in zip(time, ind):
    spikes.append(preprocess_spikes(t, i, 2))

max_spikes = 4
latest_spike_time = 25
print(f"Max spikes {max_spikes}, latest spike time {latest_spike_time}")

w_in_hid = [[4, 0, 4, 0],
            [0, 4, 0, 4 ]]
w_hid_out = [[4, 0],
             [4, 0],
             [0, 4],
             [0, 4]]

num_input = 2
num_hidden = 4
num_output = 2

rng = np.random.default_rng(1234)

network = Network()
with network:
    # Populations
    input = Population(SpikeInput(max_spikes=BATCH_SIZE * max_spikes),
                       num_input)
    hidden = Population(LeakyIntegrateFire(tau_mem=10.0), num_hidden, record_spikes=True)
    output = Population(LeakyIntegrate(tau_mem=10.0, readout="max_var"),
                        num_output)

    in_hid = Connection(input, hidden, Dense(w_in_hid), Exponential(5))
    hid_out = Connection(hidden, output, Dense(w_hid_out), Exponential(5))
max_example_timesteps = int(TRIAL_T / DT)
compiler = EventPropCompiler(example_timesteps=max_example_timesteps,
                                losses="sparse_categorical_crossentropy",
                                reg_lambda=0,
                                reg_nu_upper=1, max_spikes=4, 
                                optimiser=Adam(0), batch_size=BATCH_SIZE)
compiled_net = compiler.compile(network, optimisers= {hidden: {"tau_mem": Adam(0.01)}})

with compiled_net:
    callbacks = [
        SpikeRecorder(hidden, key="spikes_hidden"),
        VarRecorder(output,"v", key="v_out")
        ]
                   
    all_tau = []
    fig, ax = plt.subplots(2,1)
    fig, ax2 = plt.subplots(2,1)
    for e in range(NUM_EPOCHS):
        metrics, cb  = compiled_net.train({input: spikes},
                                          {output: labels}, shuffle= False,
                                          start_epoch=e, num_epochs=1, callbacks=callbacks)
        tau = compiled_net.genn_model.neuron_populations["Pop1"].vars["tau_mem"]
        tau.pull_from_device()
        all_tau.append(np.copy(tau.view))
        if e == 0 or e == NUM_EPOCHS-1:
            t= cb["spikes_hidden"][0]
            id = cb["spikes_hidden"][1]
            ax[0].scatter(t[0],id[0],s=1)
            ax[1].scatter(t[1],id[1],s=1)
            print(t[0])
            print(id[0])
            print(t[1])
            print(id[1])
            v= cb["v_out"]
            ax2[0].plot(v[0])
            ax2[1].plot(v[1])            
            print(tau.view)

    all_tau= np.asarray(all_tau)
    plt.figure()
    for i in range(4):
        plt.plot(all_tau[:,i])
    plt.show()
    
    print(f"Accuracy = {100 * metrics[output].result}%")
