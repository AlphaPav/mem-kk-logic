config="vllm"
arch="meta-llama/Meta-Llama-3-8B"

max_token=2048

models=(
 # add "YOUR_FINETUNED_MODEL_PATH"
    "ftllama/3ppl-direct-FT-50ep"
)
num_limit=100 # remove --num_limit if you want to evaluate on full dataset

for eval_nppl in 2 3 4 5 6 7 8;
do
    echo "Processing eval_nppl: $eval_nppl"
    
    for model in "${models[@]}"; 
    do
        echo "Processing model: $model"
        python eval_kk.py --batch_size 8 --model ${model} --max_token ${max_token} --arch ${model} --ntrain 0 --config ${config} --use_vllm  --limit ${num_limit} --split "train" --problem_type "clean" --eval_nppl ${eval_nppl}
        python eval_kk.py --batch_size 8 --model ${model} --max_token ${max_token} --arch ${model} --ntrain 0 --config ${config} --use_vllm  --cot --limit ${num_limit} --split "train" --problem_type "clean" --eval_nppl ${eval_nppl}
    done
done 

