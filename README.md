<h1 align="center">
<p> MareNostrum5 Distributed PyTorch Hands-On 
</h1>

- [Introduction](#introduction)
- [Singularity containers](#singularity-containers)
  - [NGC Registry](#ngc-registry)
  - [Building Docker image](#building-docker-image)
  - [Building Singularity container](#building-singularity-container)
- [First steps w/ Singularity](#first-steps-w-singularity)
    - [BSC Singularity](#bsc-singularity)
- [Slurm](#slurm)
  - [Interactive session w/ Singularity](#interactive-session-w-singularity)
  - [Batched execution w/ Singularity](#batched-execution-w-singularity)
  - [Utilitites](#utilitites)
- [torchrun](#torchrun)
- [First distributed application with Singularity](#first-distributed-application-with-singularity)
- [NCCL network benchmark](#nccl-network-benchmark)
- [Other resources](#other-resources)


## Introduction
This repository serves as an entry point for developing distributed applications with PyTorch on MareNostrum5 using Singularity containers.
## Singularity containers
### NGC Registry
The [NGC Registry](https://catalog.ngc.nvidia.com/containers) contains Docker images packed with nearly everything you can need for running applications with NVIDIA GPUs. We will use the [PyTorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) images which include all the PyTorch stack and then we will install additional dependencies. I recommend start developing from this images rather than trying to build a image from scratch (Installing CUDA drivers, NCCL, etc.)
### Building Docker image
In [`singularity/Dockerfile`](singularity/Dockerfile) you'll find a very simple Dockerfile to build a custom image from the NGC PyTorch container. We just simply copy this repository inside the `/workspace` directory of the image and install some extra dependencies. To build our custom image just run:
```sh
docker build -t mn5 -f singularity/Dockerfile .
```
The `-t` flag is the pseudonim for the builded image and `-f` to specify the path to the `Dockerfile`. Some useful flags are `--progress=plain` for a verbose installation (For debugging errors) and  `--no-cache` to perform the building process rebuilding all layers. You can check inside the container running the following:
```
docker run --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm mn5 bash
```
### Building Singularity container
> [!TIP]
> Singularity is available from MN5 General purpose and accelerated partition. Remember to load the module `module load singularity`
To build the Singularity container, we will need to compress the Docker image into a tar file and then copy it to MN5.
"
```
docker save docker_img_ID -o MN5-NGC-PyTorch-24.03.tar 
```
Then, we will request an interactive session on MN5 in the general purpose partition to create the container from the tar file as follows:
```
salloc -q gp_debug -A bsc98 --exclusive
module load singularity
singularity build MN5-NGC-PyTorch-24.03.sif docker-archive:///home/upc/upc580327/MN5-Distributed-PyTorch/MN5-NGC-PyTorch-24.03.tar
```
## First steps w/ Singularity
There are 2 ways to run applications within the container:
1. `exec`: With `exec` you'll run the specified command.
  ```
  singularity exec --nv MN5-NGC-PyTorch-24.03.sif 'nvidia-smi'
  ```
2. `shell`: With `shell` you'll run a shell inside the container:
  ```
  singularity shell --nv MN5-NGC-PyTorch-24.03.sif
  ```

> [!NOTE]
> To attach GPUs to the container add the `--nv` flag to your singularity call: `singularity shell --nv MN5-NGC-PyTorch-24.03.sif`

It is possible to bind paths inside the container with the `--bind` flag.
```
singularity shell --nv --bind /home/upc/upc580327/MN5-Distributed-PyTorch MN5-NGC-PyTorch-24.03.sif
```
#### BSC Singularity
> [!CAUTION]
> As of writing this doc, `bsc_singularity` is not available from UPC accounts. 
BSC offers a simple wrapper that allows users to list the images built by the support team called `bsc_singularity`. To list the available containers run:
```
bsc_singularity ls
  <container_1>
  <container_2>
  <container_3>
```
To use this containers you just have to switch your calls from `singularity` to `bsc_singularity`:
```
bsc_singularity exec <options> <container_X> <command>
bsc_singularity shell <options> <container_X>
```
Additionally, there's an option to print an information file that contains basic information about the container. But it may not be available for all of them.
```
bsc_singularity info <container_X>
```
## Slurm
### Interactive session w/ Singularity
For developing, interactive sessions is the way to go. You'll get a shell with the specified requirements you asked for. From this allocated hardware you'll be able to test your code and also run a container. To get a shell inside a singularity container run:
```
salloc -q acc_debug --account bsc98 --gres=gpu:2 --cpus-per-task 40 bash -c 'module load singularity && singularity shell --nv --bind /home/upc/upc580327/MN5-Distributed-PyTorch MN5-NGC-PyTorch-24.03.sif'
```
### Batched execution w/ Singularity
To submit jobs, we will use `sbatch` to send the job to a queue along with the compute requirements, and it will execute once they are available. You can check the available queues running `bsc_queues`. The script we will submit with sbatch will have a structure similar to the following snippet:
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

In [`/my_first_distributed_app/submit_my_first_distributed_app.sh`](/my_first_distributed_app/submit_my_first_distributed_app.sh), you will find an example script for launching jobs on a **single node**, and in [`/benchmark/submit_benchmark.sh`](/benchmark/submit_benchmark.sh), you will find a script for launching jobs on **multiple nodes**.
### Utilitites
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
2. Init the communications between ALL the processes
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
## First distributed application with Singularity
To test everything, I've included a small script with an application that computes PI using the trapezoid method. For this, it will divide the integral calculation among four processes and aggregate the result. The script `my_first_distributed_app/my_first_distributed_app.sh` submits the job to the accelerated partition queue and stores the output in `my_first_distributed_app/reports/`.

> [!NOTE]
> In this example, for aggregating the result we use the `gloo` backend, designed for CPU <-> CPU communications. For applications that require the use of GPUs, the `nccl` backend will be used
## NCCL network benchmark
In [`benchmark/`](/benchmark/), you'll find both the code and the scripts to launch a test that measures the bandwidth between GPUs across multiple nodes. The value we are interested in is the bus bandwidth, as it reflects [how optimally the hardware is used](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#bus-bandwidth).
| Nodes | Bus bandwidth | Algorithm bandwidth |
|:-----:|:-------------:|:-------------------:|
|   1   |     2628.8    |        1752.5       |
|   2   |     1230.8    |        703.3        |
|   4   |     656.0     |        349.9        |
|   8   |     657.9     |        339.6        |
|   16  |     660.4     |        335.4        |

## Other resources
- [BSC MareNostrum5 User Guide](https://www.bsc.es/supportkc/docs/MareNostrum5/intro)
- [BSC Singularity User Guide](https://www.bsc.es/supportkc/docs-utilities/singularity)