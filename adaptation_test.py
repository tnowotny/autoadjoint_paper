import numpy as np


from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder, Callback, VarRecorder
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import OneToOne, Dense
from ml_genn.neurons import LeakyIntegrate, SpikeInput, UserNeuron
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential


from ml_genn.utils.data import preprocess_spikes

spikes = []
labels = []
rate = 50
T = 1000
for e in range(32):
    ind = []
    time = []
    for t in range(T):
        rand_num = np.random.rand(1)
        if rand_num[0]<rate/T:
            ind.append(0)
            time.append(t)
    spikes.append(preprocess_spikes(np.array(time), np.array(ind), 1))
    labels.append(1)

AdaptiveLeakyIntegrateFire = UserNeuron(
    vars={
        "v": ("(Isyn - v)/tau_mem", "0.0"),
        "A": ("(-A)/tau_A", "A + dA")},
    threshold="v - thresh - A",
    output_var_name="v",
    param_vals={"tau_mem": 20,
                "tau_A": 200,
                "thresh": 1.0,
                "dA": 0.2
    },
    var_vals={"v": 0.0, "A": 0.0}, 
    solver="exponential_euler"
)
network = Network()
with network:
    input = Population(SpikeInput(max_spikes=32 * 200),
                       1)
    hidden = Population(AdaptiveLeakyIntegrateFire,
                        2, record_spikes=True)
    output = Population(LeakyIntegrate(readout="sum_var"),
                        2)
    
    # Connections
    inhid = Connection(input, hidden, Dense(weight=1.0),
                       Exponential(5.0))
    hidout = Connection(hidden, output, OneToOne(weight=1.0),
                       Exponential(5.0))
    
compiler = EventPropCompiler(example_timesteps=1000,
                             losses="sparse_categorical_crossentropy",
                              max_spikes=1500, 
                             batch_size=32)

optimisers= {hidden: {"tau_A": Adam(0.1)}}
compiled_net = compiler.compile(network,
                               optimisers=optimisers)

epoch = 500
with compiled_net:
    for e in range(epoch):
        metrics, cb_data  = compiled_net.train({input: spikes},
                                                {output: labels},
                                                num_epochs=1, start_epoch=e,shuffle=False,
                                                callbacks=["batch_progress_bar", SpikeRecorder(hidden, "hidden_spikes", record_counts=True),VarRecorder(hidden, var="tau_AGradient", key="tau_AGradient")])
        hidden_spikes = np.zeros(2)
        for spk in cb_data["hidden_spikes"]:
            hidden_spikes += spk
        grad = np.zeros(2)
        for spk in cb_data["tau_AGradient"]:
            grad += spk.sum(0)
        
        print(hidden_spikes, grad)            
