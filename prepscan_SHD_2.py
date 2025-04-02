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
    "NAME": "scan",
    "OUT_DIR": "scan_SHD_2",
    "SEED": 345
}

reg_lambda= [ 1e-12, 2e-12, 5e-12, 1e-11, 2e-11, 5e-11, 1e-10, 2e-10, 5e-10, 1e-9 ]


for i in range(10):
    for j in range(4):
        id = i*4+j
        print(id)
        p = p0
        p["REG_LAMBDA"] = reg_lambda[i]
        p["SEED"] = p0["SEED"] + j*11
        p["NAME"] = "J2_"+str(id)
        with open(os.path.join(p["OUT_DIR"], p["NAME"]+'.json'), 'w') as f:
            json.dump(p, f)
