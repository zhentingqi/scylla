# Licensed under the MIT license.

from modules.Task import *

import os, json
from argparse import ArgumentParser


def main(args):
    task = eval(args.task_name)()
    
    distrib_probing_save_dir = os.path.join(args.distrib_probing_save_root, str(task))
    if not os.path.exists(distrib_probing_save_dir):
        print(f"[0_5_aggregate_across_seeds.py:main] ==> {distrib_probing_save_dir} does not exist.")
        return
    
    all_outputs_across_seeds = {
        "prompt": None,
        "model_outputs": []
    }
    for js_file in os.listdir(distrib_probing_save_dir):
        if not js_file.startswith("seed") and js_file.endswith(".json"):
            continue
        seed_file_path = os.path.join(distrib_probing_save_dir, js_file)
        with open(seed_file_path, "r") as f:
            js = json.load(f)
        prompt = js["prompt"]
        model_output = js["model_output"]
        if all_outputs_across_seeds["prompt"] is None:
            all_outputs_across_seeds["prompt"] = prompt
        all_outputs_across_seeds["model_outputs"].extend(model_output)
        
    output_file = os.path.join(distrib_probing_save_dir, "all_outputs_across_seeds.json")
    with open(output_file, "w") as f:
        json.dump(all_outputs_across_seeds, f, ensure_ascii=False)


if __name__ == "__main__":
    print("********************** 0_5_aggregate_across_seeds.py **********************")
    
    parser = ArgumentParser()
    parser.add_argument("--out_root", type=str, default="out")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()
    
    args.model_name = args.model_ckpt.split("/")[-1]
    args.distrib_probing_save_root = os.path.join(args.out_root, "distrib_probing", args.model_name)
    os.makedirs(args.distrib_probing_save_root, exist_ok=True)
    
    main(args)