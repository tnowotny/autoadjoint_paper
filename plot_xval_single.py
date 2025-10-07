import numpy as np
import matplotlib.pyplot as plt
import sys

d = []
labels = []
epochs = []
for k in range(len(sys.argv)-1):
    d.append(np.loadtxt(sys.argv[k+1]))
    epochs.append(int(np.max(d[k][:,1]))+1)
    
    with open(sys.argv[k+1]) as f:
        labels.append(f.readline().strip("\n").split(" "))
        labels[k].pop(0)
colN = [ d[i].shape[1] for i in range(len(d)) ]

plotN = min(colN)
#figure out what epochs were calculated
max_epoch = max(epochs)
x_axis = []
for k in range(len(d)):
    base = np.ones(epochs[k])
    steps = np.arange(0,max_epoch*10,max_epoch)
    x_axis.append( np.outer(base,steps).T.flatten())

wd = int(np.sqrt(plotN))+1
ht = plotN // wd +1

fig, ax = plt.subplots(ht, wd, sharex=True)

plotx = []
splits = []
for k in range (len(d)):
    split = np.where(np.diff(d[k][:,1]) < 0)
    x = [0]+list(split[0])+[len(d[k][:,1])-1]
    print(f"x: {x}")
    the_x = []
    for i in range(len(x)-1):
        the_x.append(np.arange(x[i+1]-x[i])+i*max_epoch)
    plotx.append(the_x)
    splits.append(x)

print(plotx)

for y in range(ht):
    for x in range(wd):
        i = y*wd+x
        if i < plotN:
            for k in range (len(d)):
                for j in range(len(splits[k])-1):
                    print(f"d[{k}][{splits[k][j]+1}:{splits[k][j+1]+1}]")
                    ax[y,x].plot(plotx[k][j],d[k][splits[k][j]+1:splits[k][j+1]+1,i],color=f"C{k}",lw=1)
                
            ax[y,x].set_title(labels[0][i])
        if i == plotN - 1:
            ax[y,x].set_ylim([0 ,1])

plt.show()
                        
