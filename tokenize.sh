source ./config_env.sh
# MYDATA=/mnt/petrelfs/zengzhiyuan.d/.cache/huggingface/hub/datasets--cerebras--SlimPajama-627B/snapshots/2d0accdd58c5d5511943ca1f5ff0e3eb5e293543/validation/chunk1/
# MYDATA="../val_data/flatten/"
MYDATA="/mnt/inspurfs/zengzhiyuan/gpt-moe/data/openorca"
folder_path=$MYDATA

VOCAB="/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model"
TOKENIZER_TYPE=SPMTokenizer
mask_tokens="5519,12015,12336,5262"


OUTPUT_PREFIX="./data/openorca/"
srun -p llm_t -n 1 --cpus-per-task 32 --gpus-per-task 0 python tools/datasets/preprocess_data_with_mask.py \
            --input ${MYDATA} \
            --output-prefix ${OUTPUT_PREFIX} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            --vocab ${VOCAB} \
            --dataset-impl mmap \
            --append-eod \
            --workers 32 \
            --mask-before-token $mask_tokens \
            --log-interval 10000
            # --content-key text

