model="openai/gpt-4o-mini-2024-07-18"
# model="YOUT-COT-FTED-MODEL-PATH"


config="vllm"
num_limit=100


max_token=2048
ntrain=0

for eval_nppl in 2 3 4 5 6 7 8;
do
    echo "Processing eval_nppl: $eval_nppl"
    for split in "train" "test";
    do
        echo "Processing split: $split"
        
        python eval_kk.py --cot --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split ${split} --problem_type "clean"  --eval_nppl ${eval_nppl}
        python eval_kk.py --cot --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split ${split} --problem_type "random_pair"   --eval_nppl ${eval_nppl} 
        python eval_kk.py --cot --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split ${split}  --problem_type "reorder_statement"  --eval_nppl ${eval_nppl}
        python eval_kk.py --cot --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split ${split}  --problem_type "uncommon_name"  --eval_nppl ${eval_nppl}
        python eval_kk.py --cot --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split ${split}  --problem_type "flip_role"  --eval_nppl ${eval_nppl}  
        python eval_kk.py --cot --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split ${split}  --problem_type "perturbed_statement"  --eval_nppl ${eval_nppl}
        python eval_kk.py --cot --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split ${split}  --problem_type "perturbed_leaf"  --eval_nppl ${eval_nppl}
    done
done


