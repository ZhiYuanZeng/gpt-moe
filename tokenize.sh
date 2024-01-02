source ./config_env.sh
MYDATA=/mnt/petrelfs/share_data/llm_data/1006_subdomain_slimpajama/train/en/
folder_path=$MYDATA

VOCAB="../MixtralKit/mistral-moe-ckpt/tokenizer.model"
TOKENIZER_TYPE=SPMTokenizer

for file in "$folder_path"/*; do
    echo $file
    echo $OUTPUT_PREFIX
    filename=$(basename "$file") 
    OUTPUT_PREFIX="./data/slimpajama/slimpajama_mistral/${filename}"
    srun -p llm_o -n 1 --cpus-per-task 64 python tools/datasets/preprocess_data.py \
                --input $file \
                --output-prefix ${OUTPUT_PREFIX} \
                --tokenizer-type ${TOKENIZER_TYPE} \
                --vocab ${VOCAB} \
                --dataset-impl mmap \
                --append-eod \
                --workers 256 \
                --content-key text &> tokenize_${filename}.out &
done

exit

