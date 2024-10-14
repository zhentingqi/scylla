# Licensed under the MIT license.

from modules.Task import *
from utils.common_utils import avg, geometric_mean
from utils.divergence_utils import *

from argparse import ArgumentParser
import json, os


def aggregate_extrinsic_and_intrinsic_metrics(intrinsic_results, extrinsic_results, K_list=[10, 30, 50, 70, 90]):
    assert len(intrinsic_results) == len(extrinsic_results), "Intrinsic and extrinsic results have different lengths."

    # From extrinsic, we get acc
    acc = avg([r["correct"] for r in extrinsic_results if r["correct"] is not None])
    second_acc = avg([r["second_correct"] for r in extrinsic_results if r["second_correct"] is not None])

    # From intrinsic, we get avg prob
    probs = []
    for entry in intrinsic_results:
        results_for_possible_outputs = entry["metrics"]
        best_result = max(results_for_possible_outputs, key=lambda x: x["prob"])
        probs.append(best_result["prob"])
    avg_prob = avg(probs)

    # From intrinsic, we get min K% probs
    K2scores = {}
    for entry in intrinsic_results:
        results_for_possible_outputs = entry["metrics"]
        best_result = max(results_for_possible_outputs, key=lambda x: x["prob"])
        suffix_probs = best_result["suffix_probs"]
        for K in K_list:
            cut = int(K / 100 * len(suffix_probs))
            if cut == 0:
                cut = 1
            mink_percent_probs = sorted(suffix_probs)[:cut]
            # Calcuate average log likelihood
            score = geometric_mean(mink_percent_probs)
            K2scores[K] = K2scores.get(K, []) + [score]

    for K in K2scores:
        K2scores[K] = avg(K2scores[K])

    res = {
        "acc": acc, 
        "second_acc": second_acc,
        "avg_prob": avg_prob,
        "K_to_memorization_scores": K2scores, 
        "K_to_acc_norm": {K: acc / K2scores[K] for K in K2scores},
        "K_to_second_acc_norm": {K: second_acc / K2scores[K] for K in K2scores}
    }

    return res


def aggregate_extrinsic_and_intrinsic_metrics_topk(intrinsic_results, extrinsic_results, topk_list=[60, 80, 100]):
    #! Only calculate the acc using top-k probabilities
    sorted_indices = sorted(
        range(len(intrinsic_results)),
        key=lambda i: max(r["prob"] for r in intrinsic_results[i]["metrics"]),
        reverse=True,
    )
    topk2res = {}
    for topk in topk_list:
        topk_indices = sorted_indices[: len(sorted_indices) * topk // 100]
        chosen_intrinsic_results = [intrinsic_results[i] for i in topk_indices]
        chosen_extrinsic_results = [extrinsic_results[i] for i in topk_indices]
        res = aggregate_extrinsic_and_intrinsic_metrics(chosen_intrinsic_results, chosen_extrinsic_results)
        topk2res[f"top{topk}"] = res
    return topk2res


def main(args):
    task = eval(args.task_name)()
    id_extrinsic_eval_results_path = os.path.join(
        args.id_data_save_root, str(task), f"all_id_eval_results---instruction---extrinsic.json"
    )
    assert os.path.exists(
        id_extrinsic_eval_results_path
    ), f"ID eval results for extrinsic metric for task {task} does not exist."
    id_intrinsic_eval_results_path = os.path.join(
        args.id_data_save_root, str(task), f"all_id_eval_results---mixed---intrinsic.json"
    )
    assert os.path.exists(
        id_intrinsic_eval_results_path
    ), f"ID eval results for intrinsic metric for task {task} does not exist."

    with open(id_extrinsic_eval_results_path, "r") as f:
        id_extrinsic_eval_results = json.load(f)
    with open(id_intrinsic_eval_results_path, "r") as f:
        id_intrinsic_eval_results = json.load(f)

    id_results = aggregate_extrinsic_and_intrinsic_metrics_topk(id_intrinsic_eval_results, id_extrinsic_eval_results)

    print(f"[7_normalize_acc.py] ==> Saving ID normalized results for task {task}...")
    id_results_save_path = os.path.join(args.measurement_save_root, str(task), f"all_id_normalized_results.json")
    with open(id_results_save_path, "w") as f:
        json.dump(id_results, f)

    ood_extrinsic_eval_results_path = os.path.join(
        args.ood_data_save_root, str(task), f"all_ood_eval_results---instruction---extrinsic.json"
    )
    assert os.path.exists(
        ood_extrinsic_eval_results_path
    ), f"OOD eval results for extrinsic metric for task {task} does not exist."
    ood_intrinsic_eval_results_path = os.path.join(
        args.ood_data_save_root, str(task), f"all_ood_eval_results---mixed---intrinsic.json"
    )
    assert os.path.exists(
        ood_intrinsic_eval_results_path
    ), f"OOD eval results for intrinsic metric for task {task} does not exist."

    with open(ood_extrinsic_eval_results_path, "r") as f:
        ood_extrinsic_eval_results = json.load(f)
    with open(ood_intrinsic_eval_results_path, "r") as f:
        ood_intrinsic_eval_results = json.load(f)

    ood_results = aggregate_extrinsic_and_intrinsic_metrics_topk(ood_intrinsic_eval_results, ood_extrinsic_eval_results)

    print(f"[7_normalize_acc.py] ==> Saving OOD normalized results for task {task}...")
    ood_results_save_path = os.path.join(args.measurement_save_root, str(task), f"all_ood_normalized_results.json")
    with open(ood_results_save_path, "w") as f:
        json.dump(ood_results, f)


if __name__ == "__main__":
    print("********************** 7_normalize_acc.py **********************")

    parser = ArgumentParser()
    parser.add_argument("--out_root", type=str, default="out")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()

    args.model_name = args.model_ckpt.split("/")[-1]
    args.id_data_save_root = os.path.join(args.out_root, "id_data", args.model_name)
    args.ood_data_save_root = os.path.join(args.out_root, "ood_data", args.model_name)
    args.measurement_save_root = os.path.join(args.out_root, "measurement", args.model_name)

    main(args)
