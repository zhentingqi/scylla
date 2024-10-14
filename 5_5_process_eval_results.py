# Licensed under the MIT license.

from modules.Task import *
from utils.common_utils import TaskType, Complexity

from argparse import ArgumentParser
import json, os
from tqdm import tqdm


def process_eval_results(args, task, eval_results_save_path):
    with open(eval_results_save_path, "r") as f:
        eval_results = json.load(f)
        
    new_results = []
    print(f"[5_5_process_eval_results.py:main] ==> Processing eval results for task {task} at {eval_results_save_path}...")
    for item in tqdm(eval_results):
        if "model_generation" in item:
            model_generation = item["model_generation"]
            desired_output_js = item["desired_output_js"]
            
            model_answer_js = task.extract_answer(model_generation)
            item["model_answer_js"] = model_answer_js
            if model_answer_js is None:
                item["correct"] = False
            else:
                item["correct"] = task.evaluate(model_answer_js, desired_output_js[0])

            second_model_answer_js = task.extract_answer(item["second_model_generation"])
            item["second_model_answer_js"] = second_model_answer_js
            if second_model_answer_js is None:
                item["second_correct"] = False
            else:
                item["second_correct"] = task.evaluate(second_model_answer_js, desired_output_js[0])
                
            new_results.append(item)
    
    if len(new_results) > 0:
        assert len(new_results) == len(eval_results), "Number of results should be the same."
        with open(eval_results_save_path, "w") as f:
            json.dump(new_results, f, ensure_ascii=False)


def main(args):
    task = eval(args.task_name)()
    id_eval_results_save_path = os.path.join(args.id_data_save_root, str(task), f"all_id_eval_results---{args.prompt_mode}---{args.metric_type}.json")
    assert os.path.exists(id_eval_results_save_path), f"ID eval results for task {task} does not exist."
    process_eval_results(args, task, id_eval_results_save_path)
    
    if task.task_type == TaskType.ARITHMETIC:
        ood_eval_results_save_path = os.path.join(args.ood_data_save_root, str(task), f"all_ood_eval_results---{args.prompt_mode}---{args.metric_type}.json")
        assert os.path.exists(ood_eval_results_save_path), f"OOD eval results for task {task} does not exist."
        process_eval_results(args, task, ood_eval_results_save_path)
    else:
        print(f"[5_5_process_eval_results.py:main] ==> OOD eval results processing is not supported for non-arithmetic tasks.")


if __name__ == '__main__':
    print("********************** 5_5_process_eval_results.py **********************")
    
    parser = ArgumentParser()
    parser.add_argument("--out_root", type=str, default="out")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--prompt_mode", type=str, choices=["instruction", "fewshot", "mixed"], required=True)
    parser.add_argument("--metric_type", type=str, choices=['intrinsic', 'extrinsic'], required=True)
    args = parser.parse_args()
    
    args.model_name = args.model_ckpt.split("/")[-1]
    args.id_data_save_root = os.path.join(args.out_root, "id_data", args.model_name)
    args.ood_data_save_root = os.path.join(args.out_root, "ood_data", args.model_name)
    
    main(args)
    