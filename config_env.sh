
LLMPLATFORM=/mnt/petrelfs/share_data/llm_env

export MKL_NUM_THREADS=16
export OMP_NUM_THREADS=16

export GCC_HOME=${LLMPLATFORM}/dep/gcc-10.2.0
export MPFR_HOME=${LLMPLATFORM}/dep/mpfr-4.1.0
export CUDA_PATH=${LLMPLATFORM}/dep/cuda-11.7

export LD_LIBRARY_PATH=${GCC_HOME}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${MPFR_HOME}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_PATH}/lib64:${CUDA_PATH}/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/moe/lib/python3.8/site-packages/torch/lib/:$LD_LIBRARY_PATH

export PATH=${GCC_HOME}/bin:$PATH
export PATH=/mnt/petrelfs/share/git-2.37.1/bin:$PATH

export CC=${GCC_HOME}/bin/gcc
export CXX=${GCC_HOME}/bin/c++

export CPLUS_INCLUDE_PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/moe/include/python3.8/:$CPLUS_INCLUDE_PATH
export C_INCLUDE_PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/moe/include/python3.8/:$C_INCLUDE_PATH
export PYTHONPATH=$PYTHONPATH:/mnt/petrelfs/zengzhiyuan.d/petrel-oss-python-sdk2/

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_API_KEY="1aceff17102bea614105d6f44b26d4c9d81c1f22"
export LD_LIBRARY_PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/moe/lib/:$LD_LIBRARY_PATH
export PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/moe/bin:$PATH