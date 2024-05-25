<h1 align="center">
<p> ?Alps? Distributed PyTorch Hands-On 
</h1>

- [Introduction](#introduction)
- [**?Creating a new Image/Container??**](#creating-a-new-imagecontainer)
- [First steps w/ ?**Singularity?**](#first-steps-w-singularity)
  - [FileSystems](#filesystems)
- [Slurm](#slurm)
  - [Interactive session w/ **?Singularity?**](#interactive-session-w-singularity)
  - [Batched execution w/ Singularity](#batched-execution-w-singularity)
  - [Slurm Queues](#slurm-queues)
  - [Utilitites](#utilitites)
- [torchrun](#torchrun)
  - [NCCL network benchmark](#nccl-network-benchmark)
- [Other resources](#other-resources)


## Introduction
This repository serves as an entry point for developing distributed applications with PyTorch on **?Alps?** using **?Singularity containers?**.

## **?Creating a new Image/Container??**
?**In this section we'll go across all the necessary steps to create a new image for..........?**

## First steps w/ ?**Singularity?**
**I don't know exactly the details, but I think that we hace a toml file with the configs right? In this section we explain how to launch the jobs with the images, bind pathes, use gpus, aws plugin, etc**
There are 2 ways to run applications within the container:
1. ......
2. ......
3. ......

Also important:
- **?How to bind paths?**
- **?How to Use the gpus?**
- **?The aws performance plugin?**

### FileSystems
Which are the different filesystems in the cluster and how we should use them. (((Is there any folder shared among the group???)))


## Slurm
### Interactive session w/ **?Singularity?**
For developing, interactive sessions is the way to go. You'll get a shell with the specified requirements you asked for. From this allocated hardware you'll be able to test your code and also run a container. To get a shell inside a singularity container run:
```
salloc -q acc_debug --account bsc98 --gres=gpu:2 --cpus-per-task 40 bash -c 'module load singularity && singularity shell --nv --bind /home/upc/upc580327/MN5-Distributed-PyTorch MN5-NGC-PyTorch-24.03.sif' **?Change this line!!! Leave a single line command to launch a shell inside a container!!?**
```
### Batched execution w/ Singularity
To submit jobs, we will use `sbatch` to send the job to a queue along with the compute requirements, and it will execute once they are available. You can check the available queues running `bsc_queues` (**?Is there a similar command in alps?**). The script we will submit with `sbatch` will have a structure similar to the following snippet:

**?Change the flags to the required by alps. It's true that with sbatch you can also run sbatch --nodes XX script.sh to set the value of the slurm variables, but let's emphasize on launching jobs with sbatch script.sh and include ALL flags inside the header of the script. ?**

**?With Alejandro we saw that there is a slum flag named #SBATCH --contigous that schedules multinode jobs with nodes close together. This is important for performance, but we experienced a limit in this flag (You could not ask for more than 32 contigous nodes I think). Check with the alps or in the docs if there is something about this flag!?**



**?Also include other relevant settings apart from the slurm flags, like environment variables for nccl if it's necessary (Ask the systems admins)?**
```
#!/bin/bash

#SBATCH --job-name batched_execution
#SBATCH --chdir /home/upc/upc580327/MN5-Distributed-PyTorch
#SBATCH --output benchmark/reports/R-%x.%j.out
#SBATCH --error benchmark/reports/R-%x.%j.err
#SBATCH --nodes 2                   
#SBATCH --ntasks-per-node 1       
#SBATCH --gres gpu:4               
#SBATCH --cpus-per-task 80         
#SBATCH --time 00:02:00            
#SBATCH --account bsc98
#SBATCH --qos acc_bsccs

...

srun --cpus-per-task $SLURM_CPUS_PER_TASK ...
```
Comments about each setting:
- `--job-name`: Job name, useful for identifyng each job.
- `--chdir`: Directory from which we will launch the job. However, it is recommended to specify absolute paths.
- `--output` and `--error`: Path to the file where we will store the `stdout` and `stderr` outputs. `%x` will return the job name, while `%j` will return the job ID.
- `--nodes`: Number of nodes requested.
- `--ntasks-per-node`: Number of Slurm tasks to be executed per node. It is worth noting that `torchrun` will represent only 1 task.
- `--gres`: The special resources we are requesting, in this case, the GPUs.
- `--cpus-per-task`: The value of this configuration should be 20 * Number of GPUs requested. 
- `--time`: The maximum execution time for the job. It is mandatory to configure it and cannot exceed the queue limit.
- `--account` and `--qos`: The account we will use to submit the job to the specified queue. Both parameters must be configured.

Remember that it is necessary to specify the `--cpus-per-task` quantity in `srun`. We recommend using the Slurm environment variable `$SLURM_CPUS_PER_TASK`.

> [!WARNING]
> Remember to always include `srun` in the `sbatch` script for the command launching `torchrun`.

### Slurm Queues
If there are multiple slurm queues, how are they used, time limits, etc. (((Usually there is a debugging queue, let's emphasize on a responsible use of it)))

### Utilitites
**?Check that all work in Alps and if there are different ones?**
- Display all submitted jobs (from all your current accounts/projects):
  ```
  squeue
  ```
- Remove a job from the queue system, canceling the execution of the processes (if they were still running):
  ```
  scancel {jobid}
  ```
- Get an estimate of when the jobs will run:
  ```
  squeue --start
  ```
## torchrun
We will use `torchrun` to run distributed applications with PyTorch. In short, `torchrun` will be in charge of:
1. Spawn `--nproc_per_node` processes in each node running the `python_script.py` file
2. Do a rendezvous between ALL the processes
3. Set the `WORLD_SIZE` and `RANK` environment variables in each process
```
LAUNCHER="torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --tee 3 \
    python_script.py
    "
```
Comments about each setting:
- `--nproc_per_node`: The number of processes we will launch on each node. For applications that use GPUs, this value should be equal to the number of GPUs per node to maintain the 1 process per GPU relationship.
- `--nnodes`: The number of nodes on which we want to run the program. It is recommended to set it using the Slurm environment variable `$SLURM_NNODES`, which will contain the number of requested nodes (`#SBATCH --nodes`).
- `--node_rank`: Rank of the node for multi-node distributed training. Although it is not mandatory for running distributed applications, it is recommended to set it using the Slurm environment variable `$SLURM_PROCID`.
- `--rdzv_endpoint`: The IP address and port of the master node to which all workers will try to connect at the start of execution to initiate communications. To set this up do the following:
  ```
  MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
  MASTER_PORT=6000
  ```
- `--rdzv_backend`: Use the `c10d` backend.
- `--tee`: Set to `3` to redirect both stdout+stderr for all workers.

This is all you need to launch multinode jobs with PyTorch in Alps. In [`submit_multinode.sh`](/slurm/submit_multinode.sh) you will find a template that includes the most important configurations. Feel free to contact ... or check slack ... for issues!

### NCCL network benchmark
In [`benchmark/`](/benchmark/), you'll find both the code and the scripts to launch a test that measures the bandwidth between GPUs across multiple nodes. The value we are interested in is the bus bandwidth, as it reflects [how optimally the hardware is used](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bus-bandwidth).
| Nodes | Bus bandwidth | Algorithm bandwidth |
|:-----:|:-------------:|:-------------------:|
|   1   |     2512.1    |        1674.7       |
|   2   |     1192.2    |        681.2        |
|   4   |     655.7     |        349.7        |
|   8   |     658.3     |        339.8        |
|   16  |      662      |        336.3        |
|   32  |     663.8     |        334.5        |
|   64  |     660.3     |        331.5        |
|  128  |     633.6     |        317.4        |

## Other resources
- Alps User Guide
- Contacts, etc.