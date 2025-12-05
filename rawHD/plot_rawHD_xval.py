import numpy as np
import matplotlib.pyplot as plt
import sys

speakers = [ 0, 1, 2, 3, 6, 7, 8 , 9, 10, 11 ]
NUM_AVG = 5
d = []
labels = []
epochs = []

d= np.loadtxt(sys.argv[1])    
with open(sys.argv[1]) as f:
    labels= f.readline().strip("\n").split(" ")
    labels.pop(0)

ls = [ '-', ":", '--','-.', (0, (1, 10)), (0, (5, 10))]
run_splits = np.where(np.logical_and(d[:-1,0] != d[0,0], d[1:,0] == d[0,0]))[0]+1
print(f"run splits: {run_splits}")
run_splits = np.hstack( [[0], run_splits, [-1]])

plotN = d.shape[1]

wd = int(np.sqrt(plotN))+1
ht = plotN // wd +1

fig, ax = plt.subplots(ht, wd, sharex=True)

all_avg = []
all_std = []
for i in range(len(run_splits)-1):
    the_d = d[run_splits[i]:run_splits[i+1]]
    fold_splits = np.where(np.diff(the_d[:,1]) < 0)[0]+1
    print(f"fold splits: {fold_splits}")
    fold_splits = np.hstack( [[0], fold_splits, [-1]])
    num_epochs = fold_splits[1]
    print(num_epochs)
    avg = np.zeros((2, len(fold_splits)-1))
    std = np.zeros((2, len(fold_splits)-1))
    for y in range(ht):
        for x in range(wd):
            id = y*wd+x
            if id < plotN:
                for j in range(len(fold_splits - 1)-1):
                    ax[y,x].plot(the_d[fold_splits[j]:fold_splits[j+1],1]+the_d[fold_splits[j],0]*num_epochs,the_d[fold_splits[j]:fold_splits[j+1],id],color=f"C{int(the_d[fold_splits[j],0])}",linestyle=ls[i])
                    if id > plotN-3:
                        which = np.where(np.asarray(speakers) == int(the_d[fold_splits[j],0]))[0]
                        print(which)
                        avg[id-plotN+2,which]= np.mean(the_d[fold_splits[j+1]-NUM_AVG:fold_splits[j+1],id])
                        std[id-plotN+2,which]= np.std(the_d[fold_splits[j+1]-NUM_AVG:fold_splits[j+1],id])
                ax[y,x].set_title(labels[id])
    all_avg.append(avg)
    all_std.append(std)
for i, (avg, std) in enumerate(zip(all_avg, all_std)):
    print(f"run {i} training accuracy:")
    print(avg[0,:])
    print(std[0,:])
    print(f"run {i} validation accuracy:")
    print(avg[1,:])
    print(std[1,:])
    print(f"Total avg training: {np.mean(avg[0,:])} +/- {np.std(avg[0,:])}, validation: {np.mean(avg[1,:])} +/- {np.std(avg[1,:])}")
x = (plotN-1) % wd
y = (plotN-1) // wd
ax[y,x].set_ylim([ 0.5, 1])
plt.show()

