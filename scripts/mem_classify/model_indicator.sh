nppl=5
epoch=50

for perturb_type in "leaf" "statement" 
do 
    for nppl in 3 5
    do 
        ft_model_path="ftllama/${nppl}ppl-direct-FT-${epoch}ep"
        base_model_path="meta-llama/Meta-Llama-3-8B"
        for model_path in ${base_model_path}  ${ft_model_path}
        do
            data_file="result_qa/ftllama/${nppl}ppl-direct-FT-${epoch}ep_0shot/vllm_token2048_train_perturbed_${perturb_type}/people${nppl}_num1000_classify.jsonl"
            echo $model_path
            echo $data_file
            python mem_cls_model.py --base_model_path ${model_path} --data_file ${data_file} 
        done
    done
done 

