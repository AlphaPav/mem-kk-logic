base_model_path="meta-llama/Meta-Llama-3-8B"  # Base model path
adapter_path=""  # Adapter path from fine-tuning
target_model_path=""  # Merged model save path
python merge.py --base_model_path $base_model_path --adapter_path $adapter_path --target_model_path $target_model_path
