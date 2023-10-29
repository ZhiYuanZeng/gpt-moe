source ./config_env.sh
export MASTER_PORT=12347
srun -p llm_t -N 4 --tasks-per-node 8 --gpus-per-task 1 python deepy.py train.py -d configs/dense/ pythia_1b_finetune.yml