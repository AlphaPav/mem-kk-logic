import copy
import os
import sys
import importlib
import pprint

module_path = os.path.abspath('.')
if not module_path in sys.path:
    sys.path.append(module_path)
import lib_kk
importlib.reload(lib_kk)
import numpy as np
import json
import os
from utils import load_jsonl,write_jsonl, init_seed

init_seed(42)

import time


def convert_int_to_str(data):
    return str(data)


def combine_train_data(data_folder,file_config, output_name):
    result_records=[]
    for config in file_config:
        file_path = os.path.join(data_folder, config[0])
        records = load_jsonl(file_path)
        print(f"Loaded {len(records)} records from {file_path}")
        if config[1] < len(records):
            records = records[:config[1]]
        result_records.extend(records)
    output_file=os.path.join(data_folder, output_name)
    write_jsonl(output_file, result_records)


def format_solution_text(ans):
    gold = ans.replace(" and ", "").replace(".", "")
    gold_conditions=gold.split(",")
    reformat_gold_conditions=[]
    for condition in gold_conditions:
        # Remove leading and trailing spaces
        gold_condition=condition.strip()
        reformat_gold_conditions.append(gold_condition)

    formatted_statements = "\n".join([f"({i+1}) {reformat_gold_conditions[i]}" for i in range(len(reformat_gold_conditions))])
    return formatted_statements


def generate_problems(n_problems, n_people, gen_perturb=True):
    problems = []
    problem_seed=1234
    start_time = time.time()
    problem_sampler = lib_kk.KKProblemSampler(problem_seed, n_people=n_people)
    problems = problem_sampler.sample_valid_problems(n_problems)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f'{len(problems)} valid problems generated')
    if gen_perturb:
        start_time = time.time()
        per_stat = problem_sampler.perturb_problems(problems, perturb_type='statement', num_perturb=1)
        perturbed_problems_statement = [item for inner_list in per_stat for item in inner_list]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f'{len(perturbed_problems_statement)} perturbed (statement) problems generated')

        start_time = time.time()
        per_stat = problem_sampler.perturb_problems(problems, perturb_type='leaf', num_perturb=1)
        perturbed_problems_leaf = [item for inner_list in per_stat for item in inner_list]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")
        print(f'{len(perturbed_problems_leaf)} perturbed (leaf) problems generated')

        return problems, perturbed_problems_statement, perturbed_problems_leaf
    else:
        return problems, None, None

def generate_wrong_problems(n_problems, n_people):
    problems = []
    problem_seed=1234
    start_time = time.time()
    problem_sampler = lib_kk.KKProblemSampler(problem_seed, n_people=n_people)
    problems = problem_sampler.sample_invalid_problems(n_problems)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    print(f'{len(problems)} valid problems with wrong answers generated')

    return problems



def generate_formatted_problem(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair,uncommon_name=False, reorder_statement=False):
    data =[]
    problem_seed=1234
    for i in range(item_start_idx, item_start_idx+ num_samples):
        problem= problems[i]
        if problem is None:
            continue

        formatter_seed= problem_seed+i
        formatter = lib_kk.KKProblemFormatter(formatter_seed, problem)
        formatted_problem = formatter.format_problem(random_knight_knave_pairs=random_knight_knave_pairs, 
                                            flip_knight_knave_pair=flip_knight_knave_pair, 
                                            random_names=True, random_saying_template=True,
                                            uncommon_name=uncommon_name, reorder_statement=reorder_statement)
        
        chain_of_thoughts = lib_kk.generate_chain_of_thoughts(problem['statements'])
        header, steps, footer = lib_kk.format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=False, repeat_claim_for_contradiction=False)
        
        repeat_header, repeat_steps, repeat_footer = lib_kk.format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=True, repeat_claim_for_contradiction=True)
        
        item= copy.deepcopy(formatted_problem)
        item["solution_text_format"]= format_solution_text(item["solution_text"])
        item["cot_head"]=header
        item["cot_repeat_steps"]=repeat_steps
        item["cot_foot"]=footer
        item["statements"]=convert_int_to_str(problem["statements"]) # convert 0/1 into "0"/"1" for future json loading
        item["index"] = i
        
        data.append(item)
    return data


def generate_data(num_samples_test, num_samples_train, num_samples_val, n_people):
    num_problems=num_samples_test+num_samples_train+num_samples_val

    clean_problems, perturbed_problems_statement, perturbed_problems_leaf = generate_problems(num_problems, n_people, gen_perturb=True)
    problems_dict={
        "clean": clean_problems,
        "perturbed_statement": perturbed_problems_statement,
        "perturbed_leaf": perturbed_problems_leaf
    }

    random_knight_knave_pairs=False
    flip_knight_knave_pair=False
    uncommon_name=False
    for problem_type, problems in problems_dict.items():
        item_start_idx=0
        for (split, num_samples) in [("test", num_samples_test), ("train", num_samples_train), ("val", num_samples_val)]:
            if num_samples==0:
                continue 
            data= generate_formatted_problem(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair, uncommon_name)

            config=f"people{n_people}_num{num_samples}"

            if random_knight_knave_pairs:
                config +="_random_pair"
            if flip_knight_knave_pair:
                config +="_flip_role"
            if uncommon_name:
                config +="_uncommon_name"
            
            output_folder=f"data/{split}/{problem_type}"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'{config}.jsonl')
            with open(output_file, 'w') as file:
                for item in data:
                    json_line = json.dumps(item)
                    file.write(json_line + '\n')
            print(f"Data has been written to {output_file}")
            item_start_idx+=num_samples

def generate_data_language_perturb(num_samples_test, num_samples_train, num_samples_val, n_people):
    num_problems=num_samples_test+num_samples_train+num_samples_val

    clean_problems, _, _ = generate_problems(num_problems, n_people, gen_perturb=False)
    problems_dict={
        "clean": clean_problems, 
    }
    perturb_list=["random_pair", "flip_role", "uncommon_name", "reorder_statement"]

    for perturb_type in perturb_list:
        random_knight_knave_pairs=False
        flip_knight_knave_pair=False
        uncommon_name=False
        reorder_statement=False
        if perturb_type=="random_pair":
            random_knight_knave_pairs=True
        elif perturb_type=="flip_role":
            flip_knight_knave_pair=True
        elif perturb_type=="uncommon_name":
            uncommon_name=True
        elif perturb_type=="reorder_statement":
            reorder_statement=True
    
        item_start_idx=0
        for (split, num_samples) in [("test", num_samples_test), ("train", num_samples_train), ("val", num_samples_val)]:
            if num_samples==0:
                continue 
            data= generate_formatted_problem(clean_problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair, uncommon_name,reorder_statement)
           
            config=f"people{n_people}_num{num_samples}"

            
            output_folder=f"data/{split}/{perturb_type}"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'{config}.jsonl')
            with open(output_file, 'w') as file:
                for item in data:
                    json_line = json.dumps(item)
                    file.write(json_line + '\n')
            print(f"Data has been written to {output_file}")
            item_start_idx+=num_samples


def generate_formatted_wrong_problem(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair,uncommon_name=False):
    data =[]
    problem_seed=1234
    for i in range(item_start_idx, item_start_idx+ num_samples):
        problem= problems[i]
        if problem is None:
            continue

        formatter_seed= problem_seed+i
        formatter = lib_kk.KKProblemFormatter(formatter_seed, problem)
        formatted_problem = formatter.format_problem(random_knight_knave_pairs=random_knight_knave_pairs, 
                                            flip_knight_knave_pair=flip_knight_knave_pair, 
                                            random_names=True, random_saying_template=True,
                                            uncommon_name=uncommon_name)
        
        item= copy.deepcopy(formatted_problem)
        item["solution_text_format"]= format_solution_text(item["solution_text"])
        item["cot_head"]="placeholder"
        item["cot_repeat_steps"]=["placeholder"]
        item["cot_foot"]="placeholder"
        item["statements"]=convert_int_to_str(problem["statements"]) # convert 0/1 into "0"/"1" for future json loading
        item["index"] = i
        
        data.append(item)
    return data


def generate_wrong_data(num_samples_test, num_samples_train, num_samples_val, n_people):
    num_problems=num_samples_test+num_samples_train+num_samples_val

    clean_problems = generate_wrong_problems(num_problems, n_people)
    problems_dict={
        "clean": clean_problems,
    }
    random_knight_knave_pairs=False
    flip_knight_knave_pair=False
    uncommon_name=False
    for problem_type, problems in problems_dict.items():
        item_start_idx=0
        for (split, num_samples) in [("test", num_samples_test), ("train", num_samples_train), ("val", num_samples_val)]:
            if num_samples==0:
                continue 
            data= generate_formatted_wrong_problem(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair, uncommon_name)

            config=f"people{n_people}_num{num_samples}"

            if random_knight_knave_pairs:
                config +="_random_pair"
            if flip_knight_knave_pair:
                config +="_flip_role"
            if uncommon_name:
                config +="_uncommon_name"
            
            output_folder=f"data/wrong/{split}/{problem_type}"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'{config}.jsonl')
            with open(output_file, 'w') as file:
                for item in data:
                    json_line = json.dumps(item)
                    file.write(json_line + '\n')
            print(f"Data has been written to {output_file}")
            item_start_idx+=num_samples




def generate_formatted_wrong_cot(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair,uncommon_name=False,  wrong_type="shuffle" ):
    data =[]
    problem_seed=1234
    for i in range(item_start_idx, item_start_idx+ num_samples):
        problem= problems[i]
        if problem is None:
            continue

        formatter_seed= problem_seed+i
        rng = np.random.default_rng(formatter_seed)
        formatter = lib_kk.KKProblemFormatter(formatter_seed, problem)
        formatted_problem = formatter.format_problem(random_knight_knave_pairs=random_knight_knave_pairs, 
                                            flip_knight_knave_pair=flip_knight_knave_pair, 
                                            random_names=True, random_saying_template=True,
                                            uncommon_name=uncommon_name)
        
        chain_of_thoughts = lib_kk.generate_chain_of_thoughts(problem['statements'])
        header, steps, footer = lib_kk.format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=False, repeat_claim_for_contradiction=False)
        
        repeat_header, repeat_steps, repeat_footer = lib_kk.format_chain_of_thoughts(problem, formatted_problem, chain_of_thoughts,
                                        repeat_claim_for_assumption=True, repeat_claim_for_contradiction=True)
        
        if wrong_type=="shuffle":
            rng.shuffle(repeat_steps)
        item= copy.deepcopy(formatted_problem)
        item["solution_text_format"]= format_solution_text(item["solution_text"])
        item["cot_head"]=header
        item["cot_repeat_steps"]=repeat_steps
        item["cot_foot"]=footer
        item["statements"]=convert_int_to_str(problem["statements"]) # convert 0/1 into "0"/"1" for future json loading
        item["index"] = i
        
        data.append(item)

    if wrong_type=="replace_one_step":
        rng = np.random.default_rng(problem_seed)
        for j , item in enumerate(data): 
            wrong_step_idx=rng.integers(0, len(item["cot_repeat_steps"]))
            original_step=item["cot_repeat_steps"][wrong_step_idx]

            possible_replacements = [i for i in range(len((data))) if j != i]
            
            while True:
                replace_item_idx=  rng.choice(possible_replacements)
                replace_item = data[replace_item_idx]
                replace_step_idx=rng.integers(0, len(replace_item["cot_repeat_steps"]))
                replace_step = replace_item["cot_repeat_steps"][replace_step_idx]
                for name_idx, name in enumerate(replace_item["names"]):
                    replace_step=replace_step.replace(name, item["names"][name_idx])
                if original_step!=replace_step:
                    item["cot_repeat_steps"][wrong_step_idx]=replace_step
                    break 

    return data


def generate_wrong_cot_data(num_samples_test, num_samples_train, num_samples_val, n_people, wrong_type="shuffle"):
    num_problems=num_samples_test+num_samples_train+num_samples_val

    clean_problems, _, _ = generate_problems(num_problems, n_people, gen_perturb=False)
    problems_dict={
        "clean": clean_problems,
    }
    random_knight_knave_pairs=False
    flip_knight_knave_pair=False
    uncommon_name=False
    for problem_type, problems in problems_dict.items():
        item_start_idx=0
        for (split, num_samples) in [("test", num_samples_test), ("train", num_samples_train), ("val", num_samples_val)]:
            if num_samples==0:
                continue 
            data= generate_formatted_wrong_cot(problems, item_start_idx, num_samples, random_knight_knave_pairs, flip_knight_knave_pair, uncommon_name, wrong_type)

            config=f"people{n_people}_num{num_samples}"

            if random_knight_knave_pairs:
                config +="_random_pair"
            if flip_knight_knave_pair:
                config +="_flip_role"
            if uncommon_name:
                config +="_uncommon_name"
            
            output_folder=f"data/wrong_cot_{wrong_type}/{split}/{problem_type}"
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f'{config}_wrong1.jsonl')
            with open(output_file, 'w') as file:
                for item in data:
                    json_line = json.dumps(item)
                    file.write(json_line + '\n')
            print(f"Data has been written to {output_file}")
            item_start_idx+=num_samples

#### main & leaf/statement perturbed generation 
for n_people in [2]:
    generate_data(num_samples_test=100,num_samples_train=200,num_samples_val=0,
                    n_people=n_people)

for n_people in [3, 4,5,6,7,8]:
    generate_data(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
                    n_people=n_people)


#### LANAGUGE perturbation
for n_people in [2]:
    generate_data_language_perturb(num_samples_test=100,num_samples_train=200,num_samples_val=0,
                    n_people=n_people)


for n_people in [3, 4,5,6,7,8]:
    generate_data_language_perturb(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
                    n_people=n_people)


# #### wrong CoT generation 
# wrong_type="replace_one_step"

# for n_people in [5]:
#     generate_wrong_cot_data(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
#                     n_people=n_people,wrong_type=wrong_type)

# wrong_type="shuffle"

# for n_people in [5]:
#     generate_wrong_cot_data(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
#                     n_people=n_people,wrong_type=wrong_type)


# #### wrong answer generation 
# for n_people in [2]:
#     generate_wrong_data(num_samples_test=100,num_samples_train=200,num_samples_val=0,
#                     n_people=n_people)
# for n_people in [3, 4, 5,6,7,8]:
#     generate_wrong_data(num_samples_test=100,num_samples_train=1000,num_samples_val=0,
#                     n_people=n_people)



