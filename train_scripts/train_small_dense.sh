export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MASTER_PORT=12347
export WANDB_API_KEY="1aceff17102bea614105d6f44b26d4c9d81c1f22"

export LD_LIBRARY_PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/shared_llm/lib/:$LD_LIBRARY_PATH
export PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/shared_llm/bin:$PATH
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
source ~/config_env.sh
srun -p llm2 -N 1 --tasks-per-node 6 --gpus-per-task 1 python deepy.py train.py -d configs dense_small.yml local_setup.yml 