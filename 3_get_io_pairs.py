# Licensed under the MIT license.

from modules.Task import *
from utils.common_utils import TaskType, Complexity

import os
from tqdm import tqdm
from argparse import ArgumentParser


def get_io_pairs(args, task, examples_path):
    d = "id" if "id" in examples_path.split("/")[-1] else "ood"
    
    # find the directory according to the examples_path
    io_pairs_save_dir = os.path.dirname(examples_path)
    io_pairs_save_path = os.path.join(io_pairs_save_dir, f"all_{d}_io_pairs.json")
    if os.path.exists(io_pairs_save_path) and not args.force:
        print(f"IO pairs for task {task} ({d}) already exist. Skipping...")
        return
    
    io_pairs = []
    
    with open(examples_path, "r") as f:
        examples = json.load(f)
        
    assert len(examples) > 0

    print(f"[3_get_io_pairs.py:main] ==> Generating IO pairs for task {task} ({d})...")
    for e in tqdm(examples):
        try:
            i, o_tuple, i_str, o_str_tuple = task.make_io_pair(e)
        except Exception as exc:
            print(f"Error: {exc}")
            breakpoint()
            task.check_example(e)
            task.make_io_pair(e)
            
        io_pair = {
            "input": i_str,
            "input_js": task.make_input_json(i),
            "output": list(o_str_tuple),
            "output_js": [task.make_output_json(o) for o in o_tuple],
        }
        io_pairs.append(io_pair)
        
    assert len(io_pairs) > 0

    print(f"Saving IO pairs to {io_pairs_save_path}")
    with open(io_pairs_save_path, "w") as f:
        json.dump(io_pairs, f, ensure_ascii=False)


def main(args):
    task = eval(args.task_name)()
    
    id_examples_path = os.path.join(args.id_data_save_root, str(task), "all_id_examples.json")
    assert os.path.exists(id_examples_path), f"ID data for task {task} does not exist."
    get_io_pairs(args, task, id_examples_path)
    
    if task.task_type == TaskType.ARITHMETIC:
        ood_examples_path = os.path.join(args.ood_data_save_root, str(task), "all_ood_examples.json")
        assert os.path.exists(ood_examples_path), f"OOD data for task {task} does not exist."
        get_io_pairs(args, task, ood_examples_path)
    else:
        print(f"[3_get_io_pairs.py:main] ==> OOD IO pairs generation is not supported for non-arithmetic tasks. Exiting...")
    

if __name__ == "__main__":
    print("********************** 3_get_io_pairs.py **********************")
    
    parser = ArgumentParser()
    parser.add_argument("--out_root", type=str, default="out")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    args.model_name = args.model_ckpt.split("/")[-1]
    args.id_data_save_root = os.path.join(args.out_root, "id_data", args.model_name)
    args.ood_data_save_root = os.path.join(args.out_root, "ood_data", args.model_name)
        
    main(args)