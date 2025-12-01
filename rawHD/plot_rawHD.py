import matplotlib.pyplot as plt
import numpy as np
import sys

speakers = [ 0, 1, 2, 3, 6, 7, 8 , 9, 10, 11 ]
d= np.loadtxt(sys.argv[1])
ls = [ '-', '--','-.']
run_splits = np.where(np.diff(d[:,0]) < 0)[0]+1
print(f"run splits: {run_splits}")
run_splits = np.hstack( [[0], run_splits, [-1]])
for i in range(len(run_splits)-1):
    #plt.figure()
    the_d = d[run_splits[i]:run_splits[i+1]]
    fold_splits = np.where(np.diff(the_d[:,1]) < 0)[0]+1
    print(f"fold splits: {fold_splits}")
    fold_splits = np.hstack( [[0], fold_splits, [-1]])
    for j in range(len(fold_splits - 1)-1):
        plt.plot(the_d[fold_splits[j]:fold_splits[j+1],1],the_d[fold_splits[j]:fold_splits[j+1],4],color=f"C{int(the_d[fold_splits[j],0])}",linestyle=ls[i])
plt.legend(speakers)

plt.figure()
for i in range(len(run_splits)-1):
    the_d = d[run_splits[i]:run_splits[i+1]]
    fold_splits = np.where(np.diff(the_d[:,1]) < 0)[0]+1
    print(f"fold splits: {fold_splits}")
    fold_splits = np.hstack( [[0], fold_splits, [-1]])
    for j in range(len(fold_splits - 1)-1):
        e = the_d[fold_splits[j]:fold_splits[j+1]-10,1]
        ld = the_d[fold_splits[j]:fold_splits[j+1],4]
        avgld = [ np.mean(ld[i:i+10]) for i in range(len(ld)-10) ]
        plt.plot(e,avgld,color=f"C{int(the_d[fold_splits[j],0])}",linestyle=ls[i])
plt.legend(speakers)
plt.show()
         
