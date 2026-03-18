import numpy as np


from ml_genn import Connection, Network, Population
from ml_genn.callbacks import SpikeRecorder, Callback, VarRecorder
from ml_genn.compilers import EventPropCompiler
from ml_genn.connectivity import OneToOne, Dense
from ml_genn.neurons import LeakyIntegrate, SpikeInput, UserNeuron, LeakyIntegrateFire
from ml_genn.optimisers import Adam
from ml_genn.serialisers import Numpy
from ml_genn.synapses import Exponential
from ml_genn.initializers import Normal


from ml_genn.utils.data import preprocess_spikes

spikes = []
labels = []
rate = 50
T = 1000


for e in range(32):
    ind, time = [], []
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
    inhid = Connection(input, hidden, Dense(weight=10.0),
                       Exponential(5.0))
    hidout = Connection(hidden, output, OneToOne(weight=1),
                       Exponential(5.0))
    
compiler = EventPropCompiler(example_timesteps=1000,
                             losses="sparse_categorical_crossentropy",
                              max_spikes=1500, 
                             batch_size=16)

optimisers= {hidden: {"tau_A": Adam(1.0)}}
compiled_net = compiler.compile(network,
                               optimisers=optimisers)
serialiser = Numpy("checkpoints_alif")

variables = ["v", "A" ,"RingWriteOffset", "RingReadOffset", "RingWriteStartOffset", "RingReadEndOffset", "BackSpike"
,"tau_AGradient"
,"tau_A"
,"LambdavAbsSum"
,"LambdavLimit"
,"LambdaAAbsSum"
,"LambdaALimit"
,"Lambdav"
,"LambdaA"
,"tsRingWriteOffset"
, "tsRingReadOffset"]
callbacks = []
for var in variables:
    callbacks.append(VarRecorder(hidden, var=var, key=var))
callbacks.append("batch_progress_bar")
epoch = 20
with compiled_net:
    for e in range(epoch):
        metrics, cb_data  = compiled_net.train({input: spikes},
                                                {output: labels},
                                                num_epochs=1, start_epoch=e,shuffle=False,
                                                callbacks = callbacks)

        compiled_net.save((str(e),), serialiser)
