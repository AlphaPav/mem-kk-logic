config="vllm"
max_token=2048
num_limit=100

echo "Processing num_limit: $num_limit"

# no \n after `### Answer:` in the prompt (--no_linebreak) for Meta-Llama-3-8B base model
model="meta-llama/Meta-Llama-3-8B"
python eval_kk.py --no_linebreak --batch_size 8 --model ${model} --max_token ${max_token} --arch ${model} --ntrain 0 --config ${config} --use_vllm  --limit ${num_limit}  --split "test" --problem_type "perturbed_leaf"
python eval_kk.py --no_linebreak --batch_size 8 --model ${model} --max_token ${max_token} --arch ${model} --ntrain 0 --config ${config} --use_vllm  --limit ${num_limit}  --split "test" --problem_type "perturbed_statement" 



models=(
"AI-MO/NuminaMath-7B-CoT"
"deepseek-ai/deepseek-math-7b-instruct"
"microsoft/Phi-3-medium-4k-instruct"
"microsoft/Phi-3-mini-4k-instruct"
)
# Iterate over each model
for model in "${models[@]}"; 
do
    echo "Processing model: $model"
    
    python eval_kk.py --batch_size 8 --model ${model} --max_token ${max_token} --arch ${model} --ntrain 0 --config ${config} --use_vllm  --limit ${num_limit}  --split "test"  --problem_type "perturbed_leaf" 
    python eval_kk.py --batch_size 8 --model ${model} --max_token ${max_token} --arch ${model} --ntrain 0 --config ${config} --use_vllm  --limit ${num_limit}  --split "test"  --problem_type "perturbed_statement" 
    
done
