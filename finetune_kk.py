import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
import wandb
from peft import LoraConfig
from torch.nn import functional as F
from datasets import load_dataset
import random
import numpy as np
from functools import partial


def init_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, response_template, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_template = response_template
        self.after_answer_losses = []
        self.before_answer_losses = []
        self.current_epoch = 0
        self.steps_per_epoch = None
        self.accumulated_steps = 0

    def train(self, resume_from_checkpoint=None, **kwargs):
        self.current_epoch = 0  # Reset epoch counter
        self.accumulated_steps = 0  # Reset accumulated steps
        return super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Find the index of "### Answer" in the input_ids
        answer_token_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        answer_token_ids = answer_token_ids[1:]

        answer_start_indices = []

        for batch_idx, input_ids in enumerate(inputs["input_ids"]):
            for i in range(len(input_ids) - len(answer_token_ids) + 1):
                if (
                    input_ids[i: i + len(answer_token_ids)].tolist()
                    == answer_token_ids
                ):
                    answer_start_indices.append((batch_idx, i))
                    break

        if not answer_start_indices:

            exit()

            return super().compute_loss(model, inputs, return_outputs)

        # Separate inputs into before and after "### Answer"
        before_inputs = {k: [] for k in inputs.keys()}
        after_inputs = {k: [] for k in inputs.keys()}

        for batch_idx, answer_start in answer_start_indices:
            for k, v in inputs.items():
                if k == "labels":
                    labels_before = v[batch_idx].clone()
                    labels_before[answer_start:] = -100
                    before_inputs[k].append(labels_before)

                    labels_after = v[batch_idx].clone()
                    labels_after[:answer_start] = -100
                    after_inputs[k].append(labels_after)
                else:
                    before_inputs[k].append(v[batch_idx])
                    after_inputs[k].append(v[batch_idx])

        # Pad the inputs
        max_before_len = max(len(seq) for seq in before_inputs["input_ids"])
        max_after_len = max(len(seq) for seq in after_inputs["input_ids"])

        def pad_and_cut(sequences, max_len, pad_value):
            return torch.stack(
                [
                    F.pad(seq[:max_len], (0, max_len - len(seq)),
                          value=pad_value)
                    for seq in sequences
                ]
            )

        for k in before_inputs:
            pad_value = 0 if k == "attention_mask" else self.tokenizer.pad_token_id
            before_inputs[k] = pad_and_cut(
                before_inputs[k], max_before_len, pad_value
            ).to(model.device)

        for k in after_inputs:
            pad_value = 0 if k == "attention_mask" else self.tokenizer.pad_token_id
            after_inputs[k] = pad_and_cut(after_inputs[k], max_after_len, pad_value).to(
                model.device
            )

        # Compute embeddings for the segment before "### Answer" without gradients
        with torch.no_grad():
            before_outputs = model(**before_inputs)
            before_loss = before_outputs.loss

        # Compute loss for the segment after "### Answer", conditioned on the segment before
        after_outputs = model(**after_inputs)
        after_loss = after_outputs.loss

        self.after_answer_losses.append(after_loss.item())
        self.before_answer_losses.append(before_loss.item())

        self.accumulated_steps += 1
        # Check if an epoch has ended
        if self.steps_per_epoch is None:
            self.steps_per_epoch = len(self.train_dataset) // (
                self.args.train_batch_size * self.args.gradient_accumulation_steps
            )

        if (
            self.accumulated_steps % self.args.gradient_accumulation_steps == 0
            and (self.accumulated_steps // self.args.gradient_accumulation_steps)
            % self.steps_per_epoch
            == 0
        ):
            self.on_epoch_end()

        if return_outputs:
            return after_loss, (before_outputs, after_outputs)
        return after_loss

    def on_epoch_end(self):
        self.current_epoch += 1
        avg_after_loss = sum(self.after_answer_losses) / \
            len(self.after_answer_losses)
        avg_before_loss = sum(self.before_answer_losses) / len(
            self.before_answer_losses
        )
        wandb.log(
            {
                "epoch_loss/avg_after_answer": avg_after_loss,
                "epoch_loss/avg_before_answer": avg_before_loss,
            },
            step=self.current_epoch * self.steps_per_epoch,
        )

        print("epoch_loss/avg_after_answer", avg_after_loss)

        self.after_answer_losses = []
        self.before_answer_losses = []


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model on K&K with PEFT."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="train/people3_num1000.jsonl",
        help="Path to the training data file.",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="test/people3_num100.jsonl",
        help="Path to the test data file.",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./out",
        help="Output directory for the fine-tuned model.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=2, help="Number of training epochs."
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=256, help="Maximum sequence length."
    )
    parser.add_argument("--logging_steps", type=int,
                        default=1, help="Logging steps.")
    parser.add_argument("--eval_steps", type=int,
                        default=2, help="eval steps.")
    parser.add_argument(
        "--save_steps",
        type=float,
        default=0,
        help="Number of updates steps before two checkpoint saves if save_strategy=steps. Should be an integer or a float in range [0,1). If smaller than 1, will be interpreted as ratio of total training steps.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        default="steps",
        help="The checkpoint save strategy to adopt during training. Possible values are: no, epoch, steps",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="bench-conta",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--wandb_key",
        default="",
        type=str,
        help="API key for W&B.",
    )
    parser.add_argument(
        "--run_name", type=str, default="kk_ft_sol_format", help="Wandb run name."
    )
    parser.add_argument("--cot_ft", action="store_true")
    parser.add_argument("--add_eos", action="store_true")

    return parser.parse_args()


# Formatting function
def formatting_prompts_func(example, eos_token):
    output_texts = []

    from dataset.prompt import system_instruction_no_reason

    for i in range(len(example["quiz"])):
        text = (
            system_instruction_no_reason
            + f"\n\n### Question: {example['quiz'][i]}\n### Answer:\nCONCLUSION:\n{example['solution_text_format'][i]}"
        )
        text += eos_token
        output_texts.append(text)
        if i == 0:
            print(text)

    return output_texts


def formatting_prompts_func_cot(example, eos_token):
    output_texts = []
    from dataset.prompt import system_instruction

    cot_head = "Let's think step by step, by considering whether each person is lying and if that leads to contradiction."
    for i in range(len(example["quiz"])):
        cot_steps = example["cot_repeat_steps"][i]
        cot_steps = " ".join(cot_steps)
        cot_foot = example["cot_foot"][i]
        text = (
            system_instruction
            + f"\n\n### Question: {example['quiz'][i]}\n### Answer: {cot_head} {cot_steps} {cot_foot}\nCONCLUSION:\n{example['solution_text_format'][i]}"
        )
        text += eos_token

        if i == 0:
            print(text)
        output_texts.append(text)
    return output_texts


def main():
    init_seed()
    args = parse_args()
    peft_config = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
    )

    # Response template and data collator
    if args.cot_ft:
        response_template = "\n### Answer: Let's think step by step"
    else:
        response_template = "\n### Answer:\n"

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize wandb

    _ = os.system("wandb login {}".format(args.wandb_key))
    os.environ["WANDB_API_KEY"] = args.wandb_key
    wandb.init(project=args.project_name, name=args.run_name)
    wandb.config.update(args)

    # Load dataset
    kk_dataset = load_dataset('K-and-K/knights-and-knaves', data_files={
        "train": [args.train_data],
        "test": [args.test_data],
    },)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_checkpoint,
        load_in_4bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    if args.add_eos:
        eos_token = tokenizer.eos_token
    else:
        eos_token = "" 
    print("eos_token", eos_token)

    new_format_func = partial(
        formatting_prompts_func_cot if args.cot_ft else formatting_prompts_func, eos_token=eos_token)

    # Initialize trainer
    trainer = CustomSFTTrainer(
        response_template=response_template,
        model=model,
        train_dataset=kk_dataset["train"],
        eval_dataset=kk_dataset["test"],
        formatting_func=new_format_func,
        args=SFTConfig(
            output_dir=args.output_dir,  # Set to None to disable saving
            report_to="wandb",
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            fp16=True,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            max_seq_length=args.max_seq_length,
            logging_strategy="steps",
            logging_steps=args.logging_steps,
            evaluation_strategy="epoch",
            eval_steps=args.eval_steps,
        ),
        peft_config=peft_config,
    )

    # Start training
    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    # Close wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
