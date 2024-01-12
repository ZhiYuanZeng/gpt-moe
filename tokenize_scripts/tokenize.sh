source ./config_env.sh
# MYDATA=/mnt/petrelfs/zengzhiyuan.d/.cache/huggingface/hub/datasets--cerebras--SlimPajama-627B/snapshots/2d0accdd58c5d5511943ca1f5ff0e3eb5e293543/validation/chunk1/
# MYDATA="../val_data/flatten/"

VOCAB="/mnt/petrelfs/share_data/yanhang/tokenizes/llama.model"
TOKENIZER_TYPE=SPMTokenizer
# mask_tokens="5519,12015,12336,5262"


# MYDATA="../proof_pile/dev/"
# OUTPUT_PREFIX="./data/proofpile/dev"
# srun -p llm_t -n 1 --cpus-per-task 64 --gpus-per-task 0 python tools/datasets/preprocess_data.py \
#             --input ${MYDATA} \
#             --output-prefix ${OUTPUT_PREFIX} \
#             --tokenizer-type ${TOKENIZER_TYPE} \
#             --vocab ${VOCAB} \
#             --dataset-impl mmap \
#             --append-eod \
#             --workers 64 \
#             --content-key content

# MYDATA="../proof_pile/test/"
# OUTPUT_PREFIX="./data/proofpile/dev"
# srun -p llm_t -n 1 --cpus-per-task 64 --gpus-per-task 0 python tools/datasets/preprocess_data.py \
#             --input ${MYDATA} \
#             --output-prefix ${OUTPUT_PREFIX} \
#             --tokenizer-type ${TOKENIZER_TYPE} \
#             --vocab ${VOCAB} \
#             --dataset-impl mmap \
#             --append-eod \
#             --workers 64 \
#             --content-key content


MYDATA="/mnt/petrelfs/share_data/feizhaoye/huggingface/dataset/proof-pile-2/algebraic-stack/train/"
OUTPUT_PREFIX="./data/proofpile2/algebraic-stack-train"
srun -p llm_o -n 1 --cpus-per-task 64 --gpus-per-task 0 python tools/datasets/preprocess_data.py \
            --input ${MYDATA} \
            --output-prefix ${OUTPUT_PREFIX} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            --vocab ${VOCAB} \
            --dataset-impl mmap \
            --append-eod \
            --workers 64 \
            --content-key content

MYDATA="/mnt/petrelfs/share_data/feizhaoye/huggingface/dataset/proof-pile-2/algebraic-stack/validation/"
OUTPUT_PREFIX="./data/proofpile2/algebraic-stack-validation"
srun -p llm_o -n 1 --cpus-per-task 64 --gpus-per-task 0 python tools/datasets/preprocess_data.py \
            --input ${MYDATA} \
            --output-prefix ${OUTPUT_PREFIX} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            --vocab ${VOCAB} \
            --dataset-impl mmap \
            --append-eod \
            --workers 64 \
            --content-key content

MYDATA="/mnt/petrelfs/share_data/feizhaoye/huggingface/dataset/proof-pile-2/algebraic-stack/test/"
OUTPUT_PREFIX="./data/proofpile2/algebraic-stack-test"
srun -p llm_o -n 1 --cpus-per-task 64 --gpus-per-task 0 python tools/datasets/preprocess_data.py \
            --input ${MYDATA} \
            --output-prefix ${OUTPUT_PREFIX} \
            --tokenizer-type ${TOKENIZER_TYPE} \
            --vocab ${VOCAB} \
            --dataset-impl mmap \
            --append-eod \
            --workers 64 \
            --content-key content