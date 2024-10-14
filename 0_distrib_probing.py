# Licensed under the MIT license.

from common.LM import Txt2TxtGenerator
from modules.Task import *

import os, json
from argparse import ArgumentParser


def main(args):
    task = eval(args.task_name)()

    distrib_probing_save_dir = os.path.join(args.distrib_probing_save_root, str(task))
    os.makedirs(distrib_probing_save_dir, exist_ok=True)
    output_file = os.path.join(distrib_probing_save_dir, f"seed{args.seed}.json")
    if os.path.exists(output_file) and not args.force:
        print(f"[0_distrib_probing.py:generate_test_inputs] ==> {output_file} already exists.")
        return

    model = Txt2TxtGenerator(args.model_ckpt, api=args.api, api_key=args.api_key, vllm_seed=args.seed, vllm_max_model_len=1536)

    print(f"[0_distrib_probing.py:generate_test_inputs] ==> Generating model outputs for task {task}...")
    prompt = task.random_test_inputs_prompt()
    model_output = model.generate(
        prompt,
        num_generations=args.num_generations,
        max_new_tokens=1024,
        min_new_tokens=16,
        temperature=0.9,
        top_p=0.95,
        top_k=60,
        repetition_penalty=1.2,
    )
    js = {"prompt": prompt, "model_output": model_output}
    with open(output_file, "w") as f:
        json.dump(js, f, ensure_ascii=False)


if __name__ == "__main__":
    print("********************** 0_distrib_probing.py **********************")

    parser = ArgumentParser()
    parser.add_argument("--out_root", type=str, default="out")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--api", type=str, default="vllm")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--num_generations", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.model_name = args.model_ckpt.split("/")[-1]
    args.distrib_probing_save_root = os.path.join(args.out_root, "distrib_probing", args.model_name)
    os.makedirs(args.distrib_probing_save_root, exist_ok=True)

    main(args)
