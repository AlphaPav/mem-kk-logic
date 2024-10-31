model="anthropic/claude-3-5-sonnet-20240620"


config="vllm"

max_token=2048
ntrain=0
num_limit=100

echo "Processing num_limit: $num_limit"
for eval_nppl in 2 3 4 5 6 7 8;
do
    echo "Processing ntrain: $ntrain"
    
    python eval_kk.py --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split "test" --eval_nppl ${eval_nppl}
    python eval_kk.py --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split "test"  --problem_type "perturbed_statement"  --eval_nppl ${eval_nppl}
    python eval_kk.py --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --limit ${num_limit}  --split "test"  --problem_type "perturbed_leaf"  --eval_nppl ${eval_nppl}
    # python eval_kk.py --batch_size 8 --model ${model} --max_token ${max_token}  --ntrain ${ntrain} --config ${config} --cot --limit ${num_limit}  --split "test" --eval_nppl ${eval_nppl}
done


