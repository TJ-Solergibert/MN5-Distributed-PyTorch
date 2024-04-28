#!/bin/bash

#SBATCH --job-name my_first_distributed_app
#SBATCH --chdir /home/upc/upc580327/MN5-Distributed-PyTorch
#SBATCH --output my_first_distributed_app/reports/R-%x.%j.out
#SBATCH --error my_first_distributed_app/reports/R-%x.%j.err
#SBATCH --ntasks-per-node 1         # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gres gpu:4                # Number of GPUs
#SBATCH --cpus-per-task 80          # number of CPUs per task. In MN5 must be Number of GPUs * 20
#SBATCH --time 00:02:00             # maximum execution time (DD-HH:MM:SS). Mandatory field in MN5
#SBATCH --account bsc98
#SBATCH --qos acc_debug
#SBATCH --hint nomultithread          

echo "START TIME: $(date)"

# auto-fail on any errors in this script
set -eo pipefail

# logging script's variables/commands for future debug needs
set -x

######################
### Set enviroment ###
######################
module purge
module load singularity

GPUS_PER_NODE=4
######################

# note that we don't want to interpolate `\$SLURM_PROCID` till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
#
# same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time
LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --tee 3 \
    "

PYTHON_FILE=/home/upc/upc580327/MN5-Distributed-PyTorch/my_first_distributed_app/my_first_distributed_app.py

export CMD="$LAUNCHER $PYTHON_FILE"

echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    "
SINGULARITY_CONTAINER=/home/upc/upc580327/MN5-Distributed-PyTorch/MN5-NGC-PyTorch-24.03.sif
SINGULARITY_ARGS=" \
    --bind /home/upc/upc580327/MN5-Distributed-PyTorch \
    $SINGULARITY_CONTAINER \
    "  

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS singularity exec --nv $SINGULARITY_ARGS bash -c "$CMD"

echo "END TIME: $(date)"