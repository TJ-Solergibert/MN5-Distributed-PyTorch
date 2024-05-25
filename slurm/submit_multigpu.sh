
This is very similar to the multinode one, we could just delete this example of multigpu and run the 
1 node jobs with the multinode torchrun launcher. ((It will work the same way, but there are more flags
in the torchrun call))


Change all of this for the alps but mantains structure of
1. Printing the start time & sets
2. Setting environment & Network
3. Set the launcher of the python job
4. srun the launcher
5. print end time

The main difference with alps will be the singulairty and module loads calls!!


#!/bin/bash

#SBATCH --job-name multigpu
#SBATCH --chdir /home/upc/upc580327/MN5-Distributed-PyTorch
#SBATCH --output reports/R-%x.%j.out
#SBATCH --error reports/R-%x.%j.err
#SBATCH --ntasks-per-node 1         # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gres gpu:4                # Number of GPUs
#SBATCH --cpus-per-task 80          # number of CPUs per task. In MN5 must be Number of GPUs * 20
#SBATCH --time 00:02:00             # maximum execution time (DD-HH:MM:SS). Mandatory field in MN5
#SBATCH --account bsc98
#SBATCH --qos acc_bsccs
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

PYTHON_FILE=/path/to/python/file/inside/container
PYTHON_ARGS=" \
    --batch_size 1024 \
    --model Llama3 \
    --precision bf16 \
    "

export CMD="$LAUNCHER $PYTHON_FILE $PYTHON_ARGS"

echo $CMD

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
SRUN_ARGS=" \
    --cpus-per-task $SLURM_CPUS_PER_TASK \
    --jobid $SLURM_JOB_ID \
    --wait 60 \
    "
SINGULARITY_CONTAINER=/path/to/singularity/.sif/file
SINGULARITY_ARGS=" \
    --bind /path/to/bind/folder \
    $SINGULARITY_CONTAINER \
    "  

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS singularity exec --nv $SINGULARITY_ARGS bash -c "$CMD"

echo "END TIME: $(date)"