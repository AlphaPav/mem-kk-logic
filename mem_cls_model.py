import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from datasets import load_dataset
import tqdm
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from peft import PeftModel
import json
import os


def merge_adapter(base_model_path, adapter_path):

    print("Loading adapter...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).cuda()

    if adapter_path != "":
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
        )

        model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaMA model activations on dataset"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="",
        help="Path to the dataset file (e.g., /path/to/dataset.jsonl)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="cls_robust_results",
        help="Path to the output JSON file to save results",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the dataset
    kk_dataset = load_dataset(
        "json",
        data_files={
            "test": [args.data_file],
        },
    )

    statements = []
    robust_metrics = []
    for i in range(len(kk_dataset["test"])):
        quiz = kk_dataset["test"]["quiz"][i]
        statements.append(quiz)
        
        metric = kk_dataset["test"]["robust_metric"][i]
        robust_metrics.append(metric)


    # Load pre-trained LLaMA model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path) # "meta-llama/Meta-Llama-3-8B"
    tokenizer.pad_token = tokenizer.eos_token

    model = merge_adapter(args.base_model_path, args.adapter_path)

    # Define a forward hook to capture MLP activations
    mlp_activations = {
        i: [] for i in range(len(model.model.layers))
    }  # One list per layer

    def get_mlp_activation_hook(layer_idx):
        def hook(module, input, output):
            mlp_activations[layer_idx].append(output.detach().cpu().numpy())

        return hook

    # Register hooks to all MLP layers in the transformer blocks
    for i, layer in enumerate(model.model.layers):
        layer.mlp.register_forward_hook(get_mlp_activation_hook(i))

    dataset = {i: [] for i in range(len(model.model.layers))}
    labels = {i: [] for i in range(len(model.model.layers))}

    # Function to process statements and capture activations
    def process_statements(statements, robust_metrics):
        for text, metric in tqdm.tqdm(zip (statements, robust_metrics)):
            from dataset.prompt import system_instruction_no_reason
            input_prompt =system_instruction_no_reason + f"\n\n### Question: {text}\n### Answer:\n"
            input_ids = tokenizer(
                input_prompt,
                return_tensors="pt",
              
            ).input_ids
            for i in range(len(model.model.layers)):
                mlp_activations[i] = []  # Reset activations for each layer

            # Run the model forward pass
            with torch.no_grad():
                _ = model(input_ids.cuda())

            # Store activations and corresponding labels
            for i in range(len(model.model.layers)):
                if mlp_activations[i]:  # Check if activations were captured
                    dataset[i].append(
                        mlp_activations[i][0]
                    )  # Use the first batch output
                    labels[i].append(metric)

    # Process statements 
    process_statements(statements, robust_metrics )  # Label 1 for correct

    # Train classifiers for each layer's activations
    classifiers = []
    accuracy_per_layer_train = []  # To store train accuracy
    accuracy_per_layer_test = []  # To store test accuracy

    results = {}  # Dictionary to store accuracy results

    # pdb.set_trace()

    # Splitting the data for each layer and training a classifier
    for i in tqdm.tqdm(range(len(model.model.layers))):
        X_layer = dataset[i]
        y_layer = labels[i]
        # Flatten the activations for the classifier
        # X_layer = [x.flatten() for x in X_layer]
        X_layer = [x.sum(axis=(0, 1)) for x in X_layer]

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_layer, y_layer, test_size=0.2, random_state=42
        )

        # import pdb
        # pdb.set_trace()
        # Initialize and train a simple logistic regression classifier
        clf = LogisticRegression(max_iter=100000)
        clf.fit(X_train, y_train)

        # Report train accuracy
        y_train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        accuracy_per_layer_train.append(train_accuracy)

        # Report test accuracy
        y_test_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        accuracy_per_layer_test.append(test_accuracy)

        train_probs = clf.predict_proba(X_train)
        test_probs = clf.predict_proba(X_test)

        train_auc= roc_auc_score(y_train, train_probs[:, 1]),
        test_auc = roc_auc_score(y_test, test_probs[:, 1]),


        classifiers.append(clf)  # Save the classifier
        # Store results for this layer
        results[f"layer_{i}"] = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "train_auc": train_auc,
            "test_auc": test_auc,
        }

        print(f"Layer {i} train accuracy: {train_accuracy:.4f}")
        print(f"Layer {i} classifier test accuracy: {test_accuracy:.4f}")

    # Save results to JSON
    if args.adapter_path != "":
        fname = (
            "-".join(args.adapter_path.split("/")[1:-1])
            .replace("_total_10ep", "")
            .replace("_total_100ep", "")
        )
    else:
        fname = args.base_model_path.split("/")[-1]
        # "base"

        if  "meta-llama/Meta-Llama-3-8B" in args.base_model_path:
            fname = "base_"
            fname +=  args.data_file.split("/")[2].replace("_0shot", "")


    if "leaf" in args.data_file:
        fname += "_leaf"
    elif "statement" in args.data_file:
        fname += "_statement"

    with open(os.path.join(args.output_file, f"sysprompt_{fname}.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
