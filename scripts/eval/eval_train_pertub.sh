config="vllm"
max_token=2048
arch="meta-llama/Meta-Llama-3-8B"
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
        
        for problem_type in "perturbed_statement" "perturbed_leaf" "random_pair" "reorder_statement" "uncommon_name" "flip_role";
        do

            python eval_kk.py --batch_size 8 --model ${model} --max_token ${max_token} --arch ${arch} --ntrain 0 --config ${config} --use_vllm  --limit ${num_limit} --split "train" --problem_type ${problem_type} --eval_nppl ${eval_nppl}
            
          
        done
    done
done 

