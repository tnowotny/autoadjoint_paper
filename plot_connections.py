import matplotlib.pyplot as plt
import numpy as np

import sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <basename>")
    exit(1)

bname = sys.argv[1]

wih = np.load(f"{bname}_Pop0_Pop1-weight.npy")
who = np.load(f"{bname}_Pop1_Pop2-weight.npy")


n_out = 2
n_hid = int(len(who)/n_out)
n_in = int(len(wih)/n_hid)

plt.figure()
plt.imshow(wih.reshape((n_in,n_hid)))

plt.figure()
plt.imshow(who.reshape((n_hid,n_out)))

plt.show()
