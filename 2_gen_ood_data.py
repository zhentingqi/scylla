# Licensed under the MIT license.

from modules.Task import *
from utils.common_utils import TaskType, Complexity

import os
from argparse import ArgumentParser


def main(args):
    task = eval(args.task_name)()
    if task.task_type == TaskType.ARITHMETIC:
        extracted_id_elements_save_path = os.path.join(args.id_data_save_root, str(task), "all_id_elements.json")
        extracted_id_examples_save_path = os.path.join(args.id_data_save_root, str(task), "all_id_examples.json")

        assert os.path.exists(extracted_id_elements_save_path) and os.path.exists(extracted_id_examples_save_path), f"ID data for task {task} does not exist."
        
        ood_data_save_dir = os.path.join(args.ood_data_save_root, str(task))
        os.makedirs(ood_data_save_dir, exist_ok=True)
        ood_elements_save_path = os.path.join(ood_data_save_dir, "all_ood_elements.json")
        ood_examples_save_path = os.path.join(ood_data_save_dir, "all_ood_examples.json")
        
        if os.path.exists(ood_elements_save_path) and os.path.exists(ood_examples_save_path) and not args.force:
            print(f"[2_gen_ood_data.py:main] ==> OOD data for task {task} already exists. Exiting...")
            return
        
        print(f"[2_gen_ood_data.py:main] ==> Generating OOD data for task {task}...")
        task.generate_ood_data(extracted_id_elements_save_path, extracted_id_examples_save_path, ood_elements_save_path, ood_examples_save_path)
    else:
        print(f"[2_gen_ood_data.py:main] ==> OOD data generation is not supported for non-arithmetic tasks. Exiting...")


if __name__ == "__main__":
    print("********************** 2_gen_ood_data.py **********************")
    
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