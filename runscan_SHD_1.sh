#! /bin/bash

# set the account
#SBATCH --account=structuretofunction

# set max wallclock time
#SBATCH --time=4:00:00

# set name of job
#SBATCH --job-name=job1

# set the number of nodes
#SBATCH --nodes=1

# set number of GPUs
#SBATCH --gres=gpu:4

# set the partition to use
#SBATCH --partition=booster

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=t.nowotny@sussex.ac.uk

# run the application
. ~/mlgenn_new_env/bin/activate
srun --exclusive -n 1 --gres=gpu:1 python shd_HPC.py scan_SHD_1/J1_$[$SLURM_ARRAY_TASK_ID*4] &
srun --exclusive -n 1 --gres=gpu:1 python shd_HPC.py scan_SHD_1/J1_$[$SLURM_ARRAY_TASK_ID*4+1] &
srun --exclusive -n 1 --gres=gpu:1 python shd_HPC.py scan_SHD_1/J1_$[$SLURM_ARRAY_TASK_ID*4+2] &
srun --exclusive -n 1 --gres=gpu:1 python shd_HPC.py scan_SHD_1/J1_$[$SLURM_ARRAY_TASK_ID*4+3] &

wait
