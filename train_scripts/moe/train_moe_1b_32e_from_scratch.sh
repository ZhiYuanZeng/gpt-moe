export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MASTER_PORT=12347
export WANDB_API_KEY="1aceff17102bea614105d6f44b26d4c9d81c1f22"

export PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/moe/bin:$PATH
unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY
source ./config_env.sh
srun -p llm_t -N 4 --tasks-per-node 8 --gpus-per-task 1 python deepy.py train.py -d configs/moe moe_1b_32e_from_scratch.yml