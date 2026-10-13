# Building and Running NCCL Tests Without a Container/Container environment

NOTE: all the steps should be executed as non root user ( except the nccl test launch step which can be done as root user)

installation of all softwares will ve done under followign path:
```
export _TOP=/home/benchmark/puneet_bench/nccl/compiled
```

## 1. Download and install HPC-X

Download page: [HPC-X | NVIDIA Developer](https://developer.nvidia.com/networking/hpc-x)

Select **2.51-CUDA12.x -> doca_ofed  --> Ubuntu 22.04 --> x86_64**:

<img width="940" height="449" alt="image" src="https://github.com/user-attachments/assets/746dd7c7-d796-4305-86b6-2ef1aec69a48" />


```bash

source $_TOP/cuda/env.src

cd $_TOP
wget https://content.mellanox.com/hpc/hpc-x/v2.51_cuda12/hpcx-v2.51-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64.tbz
tar -xjf hpcx-v2.51-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64.tbz

cd $_TOP/hpcx-v2.51-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64
export HPCX_HOME=$PWD
source $HPCX_HOME/hpcx-init.sh
hpcx_load
which mpirun
```

### Check the UCX transports and devices

```bash
ucx_info -d | grep -E "Transport|Device"
```

```text
# Transport: self
# Device: memory
# Transport: tcp
# Device: bond0
# Transport: tcp
# Device: eno1
# Transport: tcp
# Device: lo
# Transport: sysv
# Device: memory
# Transport: posix
# Device: memory
# Transport: rc_verbs
# Device: mlx5_0:1
# Transport: ud_verbs
# Device: mlx5_0:1
# Transport: rc_verbs
# Device: mlx5_1:1
# Transport: ud_verbs
# Device: mlx5_1:1
# Transport: rc_verbs
# Device: mlx5_4:1
# Transport: ud_verbs
# Device: mlx5_4:1
# Transport: rc_verbs
# Device: mlx5_5:1
# Transport: ud_verbs
# Device: mlx5_5:1
# Transport: rc_verbs
# Device: mlx5_6:1
# Transport: ud_verbs
# Device: mlx5_6:1
# Transport: rc_verbs
# Device: mlx5_7:1
# Transport: ud_verbs
# Device: mlx5_7:1
# Transport: rc_verbs
# Device: mlx5_8:1
# Transport: ud_verbs
# Device: mlx5_8:1
# Transport: rc_verbs
# Device: mlx5_9:1
# Transport: ud_verbs
# Device: mlx5_9:1
# Transport: rc_verbs
# Device: mlx5_bond_0:1
# Transport: ud_verbs
# Device: mlx5_bond_0:1
# Transport: cma
# Device: memory
# Transport: cuda_copy
# Device: cuda
# Transport: cuda_ipc
# Device: cuda
# Transport: gdr_copy
# Device: cuda
```

## 2. Build the NCCL libraries

```bash
cd $_TOP

# 1. Get the source and pick a version
git clone https://github.com/NVIDIA/nccl.git
cd nccl
git tag -l | sort -V | tail -15    # see available versions
git checkout v2.32.3-1

# 2. Build (H100 = compute capability 9.0)
make -j src.build CUDA_HOME=$_TOP/cuda/12.8.0  NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" BUILDDIR=$_TOP/nccllib-build
```

Clean up the library path so the system MPI under `/usr/mpi` is not picked up, and check that `libmpi` resolves PMIx/OPAL from HPC-X:

```bash
unset LD_RUN_PATH
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "^/usr/mpi" | paste -sd:)
ldd $HPCX_HOME/ompi5/lib/libmpi.so | grep -E "pmix|opal"

export LD_LIBRARY_PATH=$_TOP/nccllib-build/lib:$LD_LIBRARY_PATH
```

## 3. Build the NCCL tests

```bash
cd $_TOP
git clone https://github.com/NVIDIA/nccl-tests.git
cd $_TOP/nccl-tests/
git tag -l | sort -V | tail -30
```

```text
...
v2.16.7
v2.16.8
...
```

```bash
git checkout v2.21.1
git describe --tags

make -j MPI=1   MPI_HOME=$_TOP/hpcx-v2.51-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64/ompi5   CUDA_HOME=$_TOP/cuda/12.8.0 NCCL_HOME=$_TOP/nccllib-build NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90" NAME_SUFFIX="_mpi"
```

## 4. Run the NCCL tests

### Load the environment

Open a new shell and load the environment:

```bash
_TOP=/home/benchmark/puneet_bench/nccl/compiled
source $_TOP/cuda/env.src
source $_TOP/hpcx-v2.51-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64/hpcx-init.sh
hpcx_load

unset LD_RUN_PATH
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "^/usr/mpi" | paste -sd:)
export LD_LIBRARY_PATH=$_TOP/nccllib-build/lib:$LD_LIBRARY_PATH
```

### Single node

SSH to the node and load the environment (this uses NVLink SHARP / NVLS):

```bash
$_TOP/nccl-tests/build/all_reduce_perf_mpi -b 1G -e 16G -f 2 -g 8
```
<img width="1312" height="503" alt="image" src="https://github.com/user-attachments/assets/196d31af-675c-4e34-a6f0-dc37ba06fec4" />



### Multi-node

Preferably run from the login / head node:

```bash
mpirun --allow-run-as-root -H node049:1,node140:1 -np 2  --map-by ppr:1:node --bind-to none  -x LD_LIBRARY_PATH -x NCCL_DEBUG=WARN -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_NVLS_ENABLE=0 -x NCCL_COLLNET_ENABLE=0 -x NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_9 -x NCCL_IB_QPS_PER_CONNECTION=2 -x NCCL_IB_SPLIT_DATA_ON_QPS=0 -x NCCL_IB_ADAPTIVE_ROUTING=1 -x NCCL_IB_SL=1 -x UCX_TLS=rc   -x CUDA_MODULE_LOADING=EAGER  $_TOP/nccl-tests/build/all_reduce_perf_mpi -b 1G -e 16G -f 2 -g 8
```

<img width="1653" height="625" alt="image" src="https://github.com/user-attachments/assets/f7b776f5-28bd-4f3a-9f13-1a7d1bf8fa02" />
