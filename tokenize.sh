source ./config_env.sh
MYDATA=/mnt/petrelfs/share_data/llm_data/1006_raw_slimpajama/jsonl/train/en/chunk1
OUTPUT_PREFIX=./data/slimpajama/slimpajama_chunk1
python tools/preprocess_data.py \
            --input ${MYDATA} \
            --output-prefix ${OUTPUT_PREFIX} \
            --tokenizer-type HFTokenizer \
            --vocab ./20B_tokenizer.json \
            --dataset-impl mmap \
            --append-eod \
            --workers 64