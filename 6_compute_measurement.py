# Licensed under the MIT license.

from modules.Task import *
from utils.common_utils import avg, measure_id_ood_gap
from utils.divergence_utils import *

from argparse import ArgumentParser
import json, os


def aggregate_intrinsic_metrics(result_dict):
    metrics = result_dict["metrics"]
    new_dict = {
        "prob": max(m["prob"] for m in metrics),
        "ppl": min(m["ppl"] for m in metrics)
    }
    return new_dict


def main(args):
    task = eval(args.task_name)()
    id_eval_results_path = os.path.join(args.id_data_save_root, str(task), f"all_id_eval_results---{args.prompt_mode}---{args.metric_type}.json")
    assert os.path.exists(id_eval_results_path), f"ID eval results for task {task} does not exist."
    
    with open(id_eval_results_path, "r") as f:
        id_eval_results = json.load(f)
    
    ood_eval_results_path = os.path.join(args.ood_data_save_root, str(task), f"all_ood_eval_results---{args.prompt_mode}---{args.metric_type}.json")
    assert os.path.exists(ood_eval_results_path), f"OOD eval results for task {task} does not exist."
        
    with open(ood_eval_results_path, "r") as f:
        ood_eval_results = json.load(f)
        
    measurement_dict = {}
    if args.metric_type == "intrinsic":
        id_eval_results = [aggregate_intrinsic_metrics(r) for r in id_eval_results]
        ood_eval_results = [aggregate_intrinsic_metrics(r) for r in ood_eval_results]
        
        for metric in ['prob', 'ppl']:
            id_metrics = [r[metric] for r in id_eval_results]
            ood_metrics = [r[metric] for r in ood_eval_results]
            kld = kl_divergence(id_metrics, ood_metrics)
            jsd = js_divergence(id_metrics, ood_metrics)
            measurement_dict[metric] = {
                "num_id_samples": len(id_metrics),
                "avg_id_metric": avg(id_metrics),
                "num_ood_samples": len(ood_metrics),
                "avg_ood_metric": avg(ood_metrics),
                "kl_divergence": kld,
                "js_divergence": jsd
            }
    elif args.metric_type == "extrinsic":
        id_correct_list = [r["correct"] for r in id_eval_results if r["correct"] is not None and r["model_answer_js"] is not None]
        if len(id_correct_list) < len(id_eval_results) * 0.1:
            id_acc = 0
        else:
            id_acc = avg(id_correct_list)
        ood_correct_list = [r["correct"] for r in ood_eval_results if r["correct"] is not None and r["model_answer_js"] is not None]
        if len(ood_correct_list) < len(ood_eval_results) * 0.1:
            ood_acc = 0
        else:
            ood_acc = avg(ood_correct_list)
        
        measurement_dict["num_id_samples"] = len(id_correct_list)
        measurement_dict["num_id_correct"] = sum(id_correct_list)    
        measurement_dict["id_acc"] = id_acc
        measurement_dict["num_ood_samples"] = len(ood_correct_list)
        measurement_dict["num_ood_correct"] = sum(ood_correct_list)
        measurement_dict["ood_acc"] = ood_acc
        acc_gap = measure_id_ood_gap(id_acc, ood_acc)
        measurement_dict["acc_diff"] = acc_gap["diff"]
        measurement_dict["acc_ratio"] = acc_gap["ratio"]
        measurement_dict["acc_relative_diff"] = acc_gap["relative_diff"]
        
        id_second_correct_list = [r["second_correct"] for r in id_eval_results if r["second_model_answer_js"] is not None]
        if len(id_second_correct_list) < len(id_eval_results) * 0.5:
            id_second_acc = 0
        else:
            id_second_acc = avg(id_second_correct_list)            
        ood_second_correct_list = [r["second_correct"] for r in ood_eval_results if r["second_model_answer_js"] is not None]
        if len(ood_second_correct_list) < len(ood_eval_results) * 0.5:
            ood_second_acc = 0
        else:
            ood_second_acc = avg(ood_second_correct_list)
        
        measurement_dict["num_id_second_samples"] = len(id_second_correct_list)
        measurement_dict["num_id_second_correct"] = sum(id_second_correct_list)
        measurement_dict["id_second_acc"] = id_second_acc
        measurement_dict["num_ood_second_samples"] = len(ood_second_correct_list)
        measurement_dict["num_ood_second_correct"] = sum(ood_second_correct_list)
        measurement_dict["ood_second_acc"] = ood_second_acc
        second_acc_gap = measure_id_ood_gap(id_second_acc, ood_second_acc)
        measurement_dict["second_acc_diff"] = second_acc_gap["diff"]
        measurement_dict["second_acc_ratio"] = second_acc_gap["ratio"]
        measurement_dict["second_acc_relative_diff"] = second_acc_gap["relative_diff"]
    
    measurement_save_dir = os.path.join(args.out_root, "measurement", args.model_name, str(task))
    os.makedirs(measurement_save_dir, exist_ok=True)
    measurement_save_path = os.path.join(measurement_save_dir, f"measurement---{args.prompt_mode}---{args.metric_type}.json")
    with open(measurement_save_path, "w") as f:
        json.dump(measurement_dict, f)
    
    print(f"[6_compute_measurement.py:main] ==> Measurement results saved to {measurement_save_path}")


if __name__ == '__main__':
    print("********************** 6_compute_measurement.py **********************")
    
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
    