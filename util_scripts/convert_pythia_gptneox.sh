export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export MASTER_PORT=12345
export WANDB_API_KEY="1aceff17102bea614105d6f44b26d4c9d81c1f22"

export LD_LIBRARY_PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/shared_llm/lib/:$LD_LIBRARY_PATH
export PATH=/mnt/petrelfs/zengzhiyuan.d/miniconda3/envs/shared_llm/bin:$PATH

source ~/config_env.sh

export MASTER_PORT=12345
srun -p llm2 -N 1 --tasks-per-node 1 --gpus-per-task 1 python tools/convert_hf_to_gptneox.py \
    --hf-model-name pythia-1b \
    --output-dir checkpoints/pythia-1b/ \
    --cache-dir hf_models/pythia-1b/ \
    --config configs/pythia/pythia_1b.yml \
    --test
