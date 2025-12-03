import numpy as np
import matplotlib.pyplot as plt
import sys

d = []
labels = []
epochs = []
for k in range(len(sys.argv)-1):
    d.append(np.loadtxt(sys.argv[k+1]))
    epochs.append(int(np.max(d[k][:,1]))+1)
    
    with open(sys.argv[k-1]) as f:
        labels.append(f.readline().strip("\n").split(" "))
        labels[k].pop(0)
colN = [ d[i].shape[1] for i in range(len(d)) ]
print(len(labels[0]))
print(colN)
plotN = min(colN)
#figure out what epochs were calculated
max_epoch = max(epochs)

wd = int(np.sqrt(plotN))+1
ht = plotN // wd +1

fig, ax = plt.subplots(ht, wd, sharex=True)

for y in range(ht):
    for x in range(wd):
        i = y*wd+x
        if i < plotN:
            for k in range (len(d)):
                ax[y,x].plot(d[k][:,i],color=f"C{k}",lw=1)
                
            ax[y,x].set_title(labels[0][i])
        if i == plotN - 1:
            ax[y,x].set_ylim([0 ,1])

print(sys.argv[1:])
ax[0,0].legend(sys.argv[1:])
plt.show()

