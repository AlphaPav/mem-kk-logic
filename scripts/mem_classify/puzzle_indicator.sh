        
input_files=(
# process data that contains the labels of Consistenly Solved v.s. non Consistenly Solved Puzzles
"result_qa/ftopenai/ppl3-1000-cot-repeat-5ep_0shot/vllm_token2048_cot_train_perturbed_leaf/people3_num1000_classify.jsonl"

)
for input_file in "${input_files[@]}"; 
    do
    echo $input_file
    for text_field in "all_fields"  "quiz"  "response" "cot_repeat_steps" 
        do
        for method in "combine" "tfidf" "bow"  "wordlength"  "charlength" 
            do
                python mem_cls_puzzle.py --method ${method} --text_field ${text_field} --input_file ${input_file} --no_balance_label
            done
        done
    done
done 

