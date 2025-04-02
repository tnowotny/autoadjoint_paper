import os
import json
import numpy as np

p0= {
    "NUM_HIDDEN": 256,
    "BATCH_SIZE": 32,
    "NUM_EPOCHS": 300,
    "REG_LAMBDA": 1e-10,
    "DT": 1.0,
    "KERNEL_PROFILING": True,
    "RECORDING": True,
    "PLOTTING": True,
    "NAME": "scan",
    "OUT_DIR": "scan_SHD_1",
    "SEED": 345
}


for i in range(10):
    for j in range(4):
        id = i*4+j
        print(id)
        p = p0
        p["REG_LAMBDA"] = reg_lambda[i]
        p["SEED"] = P0["SEED"] + j*11
        p["NAME"] = "J1_"+str(id)
        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
            json.dump(p, f)
