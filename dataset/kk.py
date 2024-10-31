import numpy as np
from .prompt import system_instruction, demonstration_2char, system_instruction_no_reason, demonstration_2char_no_reason


def num_tokens_from_string(string):
    import tiktoken
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def parse_cot_eval(pred_str, ans,
                   conclusion_patterns=['CONCLUSION:'],
                   verbose=False,
                   finish_patterns=["### Reason", "Let's think step by step again", "let's go back and check", "###"],
                   reformat_gold_conditions=None):
    
    def judge_string(input_str, reformat_gold_conditions, wrong_reason, finish_patterns):
        correct_count = 0
        is_correct = False
        beyond_id = len(reformat_gold_conditions)+1
        beyond_id_pattern = f"({beyond_id})"

        for finish_pattern in finish_patterns:
            if finish_pattern in input_str:
                input_str = input_str.split(finish_pattern)[0]

        if beyond_id_pattern in input_str:
            is_correct = False
            wrong_reason = "beyond_list"
        elif "if" in input_str:
            is_correct = False
            wrong_reason = "contain_if"
        else:
            is_correct = True
            for gold_condition in reformat_gold_conditions:
                if gold_condition not in input_str:
                    is_correct = False
                    wrong_reason = "wrong_identity"
                else:
                    correct_count += 1
        correct_ratio = correct_count/len(reformat_gold_conditions)

        return is_correct, wrong_reason, correct_ratio

    def check_numbers_in_string(s, N):
        for i in range(1, N + 1):
            if f"({i})" not in s:
                return False
        return True
    
    original_str = pred_str
    pred_str = pred_str.split("### Question")[0]
    pred_answer = pred_str
    is_correct = False
    correct_ratio = 0
    if reformat_gold_conditions is None:
        gold = ans.replace(" and ", "").replace(".", "")
        gold_conditions = gold.split(",")
        reformat_gold_conditions = []
        for condition in gold_conditions:
            gold_condition = condition.strip()    # Remove leading and trailing spaces
            reformat_gold_conditions.append(gold_condition)

    wrong_reason = "no_conclusion_matched"
    for pattern in conclusion_patterns:
        pred = pred_str.split(pattern)
        if len(pred) > 1:
            if len(pred[1]) > 0:  # if the matched the answer is not empty
                pred_answer = pred[1]
                is_correct, wrong_reason, correct_ratio = judge_string(
                    pred_answer, reformat_gold_conditions, wrong_reason, finish_patterns)
                break
    if is_correct == False and wrong_reason == "no_conclusion_matched": 
        if check_numbers_in_string(pred_str, len(reformat_gold_conditions)): # the answer contains (1)..(2)..
            is_correct, wrong_reason, correct_ratio = judge_string(
                pred_str, reformat_gold_conditions, wrong_reason, finish_patterns)
    if is_correct == False and verbose == True:
        print("wrong_reason:",wrong_reason)
        print("********* \nprediction before parse:\n", original_str)
        print("********* \nprediction after parse:\n", pred_answer)

    return is_correct, pred_answer, wrong_reason, correct_ratio, reformat_gold_conditions


class KKProcessor:
    def __init__(self, cot=True, no_linebreak=True):
        self.cot = cot
        self.no_linebreak = no_linebreak

    def format_example(self, test_records, idx, model_name=None):
       
        item = test_records[idx]

        prompt = "### Question: "+item["quiz"] + "\n"
        if self.cot:
            if model_name in ["deepseek-ai/deepseek-math-7b-instruct", "AI-MO/NuminaMath-7B-CoT"]:
                prompt += "Please reason step by step, and put your final answer within \\boxed{}."
            else:
                prompt += "### Answer: Let's think step by step"
        else:
            if self.no_linebreak:
                prompt += "### Answer:"
            else:
                prompt += "### Answer:\n"
        answer = item["solution_text"]
        return prompt, answer

    def gen_test_prompt(self, ntrain, test_records, idx, model_name=None):
        if self.cot:
            train_prompt = system_instruction
        else:
            train_prompt = system_instruction_no_reason

        if ntrain == 1:
            if self.cot:
                train_prompt += "\n\n"+demonstration_2char
            else:
                train_prompt += "\n\n"+demonstration_2char_no_reason
        elif ntrain > 1:
            raise NotImplementedError

        prompt_end, answer = self.format_example(test_records, idx, model_name)
        prompt = train_prompt + "\n\n" + prompt_end

        return prompt, answer

    def _parse_cot_eval(self, pred_str, ans, model_name=None):
        conclusion_patterns = ['CONCLUSION:', 'Conclusion:', 'conclusion:']

        if model_name in ["deepseek-ai/deepseek-math-7b-instruct", "AI-MO/NuminaMath-7B-CoT"]:
            conclusion_patterns = ['boxed{', 'CONCLUSION:', 'Conclusion:', 'conclusion:']

        is_correct, pred_answer, wrong_reason, correct_ratio, reformat_gold_conditions = parse_cot_eval(
            pred_str, ans, conclusion_patterns=conclusion_patterns, verbose=False)

        return is_correct, pred_answer, reformat_gold_conditions
