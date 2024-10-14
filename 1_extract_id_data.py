# Licensed under the MIT license.

from modules.Task import *
from utils.extract_utils import extract_elements_and_examples

import os, json
from argparse import ArgumentParser
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed


num_workers = mp.cpu_count() // 4 * 3


# Function to process each item
def process_item(item, task):
    return extract_elements_and_examples(item, task)


def main(args):
    task = eval(args.task_name)()
    
    distrib_probing_save_dir = os.path.join(args.distrib_probing_save_root, str(task))
    if not os.path.exists(distrib_probing_save_dir):
        print(f"[1_extract_id_data.py:main] ==> Distrib. probing data directory {distrib_probing_save_dir} does not exist. Exiting...")
        return
    distrib_probing_outputs_across_seeds_file = os.path.join(distrib_probing_save_dir, "all_outputs_across_seeds.json")
    if not os.path.exists(distrib_probing_outputs_across_seeds_file):
        print(f"[1_extract_id_data.py:main] ==> {distrib_probing_outputs_across_seeds_file} does not exist. Exiting...")
        return
    
    with open(distrib_probing_outputs_across_seeds_file, "r") as f:
        distrib_probing_outputs_across_seeds = json.load(f)["model_outputs"]
    
    extracted_id_elements_save_dir = os.path.join(args.id_data_save_root, str(task))
    os.makedirs(extracted_id_elements_save_dir, exist_ok=True)
    extracted_id_elements_save_path = os.path.join(extracted_id_elements_save_dir, "all_id_elements.json")
    extracted_id_examples_save_dir = os.path.join(args.id_data_save_root, str(task))
    os.makedirs(extracted_id_examples_save_dir, exist_ok=True)
    extracted_id_examples_save_path = os.path.join(extracted_id_examples_save_dir, "all_id_examples.json")
    
    if os.path.exists(extracted_id_elements_save_path) and os.path.exists(extracted_id_examples_save_path) and not args.force:
        print(f"[1_extract_id_data.py:main] ==> ID data for task {task} already exists. Exiting...")
        return
    
    all_id_elements = []
    all_id_examples = []

    print(f"[1_extract_id_data.py:main] ==> Extracting ID data for task {task}...")

    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_item, item, task) for item in distrib_probing_outputs_across_seeds]
        for future in tqdm(as_completed(futures), total=len(futures)):
            results.append(future.result())
        
    for id_elements, id_examples in results:
        all_id_elements.extend(id_elements)
        all_id_examples.extend(id_examples)
    
    assert len(all_id_elements) > 0
    assert len(all_id_examples) > 0      
    
    print(f"[1_extract_id_data.py:main] ==> Extracted {len(all_id_elements)} ID elements and {len(all_id_examples)} ID examples for task {task}.")
    
    # Deduplicate
    print(f"[1_extract_id_data.py:main] ==> Deduplicating ID data for task {task}...")
    all_id_examples_str_list = [json.dumps(x) for x in all_id_examples]
    all_id_examples_str_list = list(set(all_id_examples_str_list))
    all_id_examples = [json.loads(x) for x in all_id_examples_str_list]
    print(f"[1_extract_id_data.py:main] ==> Deduplicated to {len(all_id_examples)} ID examples for task {task}.")
    
    print(f"[1_extract_id_data.py:main] ==> Saving ID data for task {task}...")
    with open(extracted_id_elements_save_path, "w") as f:
        json.dump(all_id_elements, f, ensure_ascii=False)
    
    with open(extracted_id_examples_save_path, "w") as f:
        json.dump(all_id_examples, f, ensure_ascii=False)


if __name__ == "__main__":
    print("********************** 1_extract_id_data.py **********************")

    parser = ArgumentParser()
    parser.add_argument("--out_root", type=str, default="out")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    args.model_name = args.model_ckpt.split("/")[-1]
    args.distrib_probing_save_root = os.path.join(args.out_root, "distrib_probing", args.model_name)
    args.id_data_save_root = os.path.join(args.out_root, "id_data", args.model_name)
    os.makedirs(args.distrib_probing_save_root, exist_ok=True)
    
    main(args)