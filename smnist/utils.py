import numpy as np
from ml_genn.callbacks import Callback
import matplotlib.pyplot as plt

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

def smnist_encode(images, timestep, levels=80, num_examples=60000):
    mx = np.max(np.asarray(images).flatten())
    thresh = np.arange(1,levels+1)/levels*mx
    times = []
    ids = []
    for img in images[:num_examples]:
        spk_t = []
        id = []
        x = np.asarray(img).flatten()
        for k in range(len(x)):
            idx = np.where(thresh < x[k])[0]
            tme = np.ones(len(idx))*k*timestep
            spk_t.extend(list(tme))
            id.extend(list(idx))
        times.append(np.asarray(spk_t))
        ids.append(np.asarray(id))
    return times, ids

def smnist_encode_maass(images, timestep, levels=80, num_examples=60000):
    mx = np.max(np.asarray(images).flatten())
    thresh = np.arange(1,levels+1)/levels*mx
    times = []
    ids = []
    for img in images[:num_examples]:
        spk_t = []
        id = []
        x = np.asarray(img).flatten()
        for k in range(len(x)-1):
            idx = np.where(np.logical_and(thresh >= x[k], thresh < x[k+1]))[0]
            tme = np.ones(len(idx))*k*timestep
            spk_t.extend(list(tme))
            id.extend(list(idx))
        spk_t.append(785.0)
        id.append(80)
        times.append(spk_t)
        ids.append(id)
    return times, ids
