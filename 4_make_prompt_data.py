# Licensed under the MIT license.

from modules.Task import *
from utils.common_utils import TaskType, Complexity

import os
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
import random


def make_prompt_data(args, task, io_pairs_path):
    d = "id" if "id" in io_pairs_path.split("/")[-1] else "ood"
    
    prompt_data_save_dir = os.path.dirname(io_pairs_path)
    prompt_data_save_path__instruction = os.path.join(prompt_data_save_dir, f"all_{d}_prompt_data---instruction.json")
    prompt_data_save_path__fewshot = os.path.join(prompt_data_save_dir, f"all_{d}_prompt_data---fewshot.json")
    prompt_data_save_path__mixed = os.path.join(prompt_data_save_dir, f"all_{d}_prompt_data---mixed.json")
    
    if all(os.path.exists(p) for p in [prompt_data_save_path__instruction, prompt_data_save_path__fewshot, prompt_data_save_path__mixed]) and not args.force:
        print(f"Prompt data for task {task} ({d}) already exists. Skipping...")
        return
    
    with open(io_pairs_path, "r") as f:
        io_pairs = json.load(f)
    
    total_num_io_pairs = args.num_shots + args.num_tests
    assert len(io_pairs) >= total_num_io_pairs, f"Number of IO pairs ({len(io_pairs)}) is less than the total number of IO pairs ({total_num_io_pairs})"
    selected_io_pairs = np.random.choice(io_pairs, total_num_io_pairs, replace=False)
    
    demo_io_pairs = selected_io_pairs[:args.num_shots]
    test_io_pairs = selected_io_pairs[args.num_shots:]
    
    #! Three reasoning modes: instruction, fewshot, mixed
    # Instruction
    instruction_text = task.task_description
    prefix__instruction = instruction_text + "\n\n"
    
    # Few-shot
    demo_text = "\n\n".join([f"Input: {p['input']}\nOutput: {random.choice(p['output'])}" for p in demo_io_pairs])
    prefix__fewshot = demo_text + "\n\n"
    
    # Mixed
    prefix__mixed = prefix__instruction + prefix__fewshot
    
    prompt_data__instruction = []
    prompt_data__fewshot = []
    prompt_data__mixed = []
    for pair_idx, test_io_pair in tqdm(enumerate(test_io_pairs), desc=f"Generating prompt data for {d}...", total=len(test_io_pairs)):
        raw_input = test_io_pair["input"]
        desired_output = test_io_pair["output"]
        desired_output_js = test_io_pair["output_js"]
        
        formatted_input = f"Input: {raw_input}\nOutput: "
        full_input__instruction = prefix__instruction + formatted_input
        full_input__fewshot = prefix__fewshot + formatted_input
        full_input__mixed = prefix__mixed + formatted_input
        
        prompt_data__instruction.append({
            "id": f"{args.model_name}-{args.task_name}-instruction-{pair_idx}",
            "raw_input": raw_input,
            "instruction": instruction_text,
            "demo": None,
            "full_input": full_input__instruction,
            "desired_output": desired_output,
            "desired_output_js": desired_output_js,
        })
        prompt_data__fewshot.append({
            "id": f"{args.model_name}-{args.task_name}-fewshot-{pair_idx}",
            "raw_input": raw_input,
            "instruction": None,
            "demo": demo_text,
            "full_input": full_input__fewshot,
            "desired_output": desired_output,
            "desired_output_js": desired_output_js,
        })
        prompt_data__mixed.append({
            "id": f"{args.model_name}-{args.task_name}-mixed-{pair_idx}",
            "raw_input": raw_input,
            "instruction": instruction_text,
            "demo": demo_text,
            "full_input": full_input__mixed,
            "desired_output": desired_output,
            "desired_output_js": desired_output_js,
        })
    
    with open(prompt_data_save_path__instruction, "w") as f:
        json.dump(prompt_data__instruction, f, ensure_ascii=False)
    with open(prompt_data_save_path__fewshot, "w") as f:
        json.dump(prompt_data__fewshot, f, ensure_ascii=False)
    with open(prompt_data_save_path__mixed, "w") as f:
        json.dump(prompt_data__mixed, f, ensure_ascii=False)


def main(args):
    task = eval(args.task_name)()
    
    id_io_pairs_path = os.path.join(args.id_data_save_root, str(task), "all_id_io_pairs.json")
    assert os.path.exists(id_io_pairs_path), f"ID IO pairs for task {task} does not exist."
    make_prompt_data(args, task, id_io_pairs_path)
    
    if task.task_type == TaskType.ARITHMETIC:
        ood_io_pairs_path = os.path.join(args.ood_data_save_root, str(task), "all_ood_io_pairs.json")
        assert os.path.exists(ood_io_pairs_path), f"OOD IO pairs for task {task} does not exist."
        make_prompt_data(args, task, ood_io_pairs_path)
    else:
        print(f"[4_make_prompt_data.py:main] ==> OOD Prompt data generation is not supported for non-arithmetic tasks. Exiting...")
        

if __name__ == "__main__":
    print("********************** 4_make_prompt_data.py **********************")

    parser = ArgumentParser()
    parser.add_argument("--out_root", type=str, default="out")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument("--num_tests", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    args.model_name = args.model_ckpt.split("/")[-1]
    args.id_data_save_root = os.path.join(args.out_root, "id_data", args.model_name)
    args.ood_data_save_root = os.path.join(args.out_root, "ood_data", args.model_name)
        
    main(args)