
source ~/config_env.sh

export MASTER_PORT=12345
srun -p llm_t -N 1 --tasks-per-node 1 --gpus-per-task 1 python tools/convert_hf_to_gptneox.py \
    --hf-model-name pythia-1b \
    --output-dir checkpoints/pythia-1b/ \
    --cache-dir hf_models/pythia-1b/ \
    --config configs/dense/pythia_1b_raw.yml \
    --test
