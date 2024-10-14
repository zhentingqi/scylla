# Licensed under the MIT license.

from common.LM import Txt2TxtGenerator
from modules.Task import *
from utils.common_utils import TaskType, Complexity

from argparse import ArgumentParser
import json, os, shutil, random
from tqdm import tqdm



def eval_model(args, model, task, prompt_data_path):
    eval_results_save_dir = os.path.dirname(prompt_data_path)
    d = "id" if "id" in prompt_data_path.split("/")[-1] else "ood"
    eval_results_save_path = os.path.join(
        eval_results_save_dir, f"all_{d}_eval_results---{args.prompt_mode}---{args.metric_type}.json"
    )

    if os.path.exists(eval_results_save_path) and not args.force:
        print(f"[5_eval_model.py:main] ==> Evaluation results for task {task} ({d}) already exist.")
        return

    cache_dir = os.path.join(eval_results_save_dir, "eval_model_cache")
    if not args.force:
        os.makedirs(cache_dir, exist_ok=True)

    print(
        f"[5_eval_model.py:main] ==> Evaluating model on task {task} ({d.upper()}) for metric type {args.metric_type}..."
    )

    with open(prompt_data_path, "r") as f:
        prompt_data = json.load(f)

    all_results = []
    for item in tqdm(prompt_data):
        id = item["id"]
        desired_output = item["desired_output"]
        desired_output_js = item["desired_output_js"]
        cache_file_path = os.path.join(cache_dir, f"{id}.json")
        if os.path.exists(cache_file_path) and not args.force:
            with open(cache_file_path, "r") as f:
                result_dict = json.load(f)
                assert result_dict 
        else:
            if args.metric_type == "intrinsic":
                full_input = item["full_input"]

                metrics = []
                if len(desired_output) > 6:  # factorial(3) = 6
                    desired_output = random.sample(desired_output, 6)

                for i, d_o in enumerate(desired_output):
                    # todo: parallel inference
                    prob, ppl, aux_dict = model.calculate_probability_and_perplexity(prefix=full_input, suffix=d_o)
                    res = {f"desired_output_{i}": d_o, "prob": prob, "ppl": ppl, **aux_dict}
                    metrics.append(res)

                result_dict = {"id": id, "full_input": full_input, "metrics": metrics}
            elif args.metric_type == "extrinsic":
                raw_input = item["raw_input"]
                instruction = item["instruction"]
                assert instruction is not None
                model_input = (
                    f"Here is a task: {instruction}\n"
                    "\n"
                    f"Solve the task with the following input:\n"
                    f"{raw_input}\n"
                    "\n"
                    f'IMPORTANT: End your response with "The answer is <ANSWER>" where you should fill <ANSWER> with your final answer and must format the final answer obeying the following rules: {task.answer_format_requirements}\n'
                    "\n"
                    "Your response:\nLet's think step by step."
                )

                model_generation = model.generate(input_text=model_input, temperature=0, max_new_tokens=2048)

                second_trigger = "Therefore, the answer is"
                second_model_generation = model.generate(
                    input_text=model_input + model_generation + second_trigger, temperature=0, max_new_tokens=256
                )
                second_model_generation = second_trigger + second_model_generation

                if args.calculate_metric_after_generation:
                    try:
                        model_answer = task.extract_answer(model_generation)
                        model_answer_js = json.dumps(model_answer, ensure_ascii=False)
                        correct = task.evaluate(model_answer_js, desired_output_js[0])
                        
                        second_model_answer = task.extract_answer(second_model_generation)
                        second_model_answer_js = json.dumps(second_model_answer, ensure_ascii=False)
                        second_correct = task.evaluate(second_model_answer_js, desired_output_js[0])
                    except Exception as e:
                        print(f"[5_eval_model.py:main] ==> Error occurred when evaluating model on {id}:")
                        print(e)
                        model_answer_js = None
                        second_model_answer_js = None
                        correct = None
                        second_correct = None
                else:
                    model_answer_js = None
                    second_model_answer_js = None
                    correct = None
                    second_correct = None

                result_dict = {
                    "id": id,
                    "model_input": model_input,
                    "desired_output_js": desired_output_js,
                    "model_generation": model_generation,
                    "model_answer_js": model_answer_js,
                    "correct": correct,
                    "second_model_generation": second_model_generation,
                    "second_model_answer_js": second_model_answer_js,
                    "second_correct": second_correct,
                }

            if not args.force:
                try:
                    with open(cache_file_path, "w") as f:
                        json.dump(result_dict, f, ensure_ascii=False)
                except:
                    pass

        all_results.append(result_dict)

    assert len(all_results) == len(prompt_data)

    print(f"[5_eval_model.py:main] ==> Saving evaluation results to {eval_results_save_path}...")
    with open(eval_results_save_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False)

    if not args.force and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


def main(args):
    if args.api == "hf":
        model = Txt2TxtGenerator(args.model_ckpt, api="hf")
    elif args.api == "vllm":
        model = Txt2TxtGenerator(args.model_ckpt, api="vllm", vllm_seed=args.seed, vllm_max_model_len=4096)
    elif args.api in ["openai", "anthropic"]:
        model = Txt2TxtGenerator(args.model_ckpt, api=args.api, api_key=args.api_key)
    elif args.api == "together":
        model = Txt2TxtGenerator(args.model_ckpt, api="together")
    else:
        raise ValueError(f"Invalid API: {args.api}")

    task = eval(args.task_name)()

    if args.metric_type == "extrinsic":
        assert args.prompt_mode == "instruction", "Extrinsic evaluation only supports instruction prompt mode."

    id_prompt_data_path = os.path.join(
        args.id_data_save_root, str(task), f"all_id_prompt_data---{args.prompt_mode}.json"
    )
    assert os.path.exists(id_prompt_data_path), f"ID prompt data for task {task} does not exist."
    eval_model(args, model, task, id_prompt_data_path)

    if task.task_type == TaskType.ARITHMETIC:
        ood_prompt_data_path = os.path.join(
            args.ood_data_save_root, str(task), f"all_ood_prompt_data---{args.prompt_mode}.json"
        )
        assert os.path.exists(ood_prompt_data_path), f"OOD prompt data for task {task} does not exist."
        eval_model(args, model, task, ood_prompt_data_path)
    else:
        print(f"[5_eval_model.py:main] ==> OOD evaluation is not supported for non-arithmetic tasks.")


if __name__ == "__main__":
    print("********************** 5_eval_model.py **********************")

    parser = ArgumentParser()
    parser.add_argument("--out_root", type=str, default="out")
    parser.add_argument("--model_ckpt", type=str, required=True)
    parser.add_argument("--api", type=str, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_mode", type=str, choices=["instruction", "fewshot", "mixed"], required=True)
    parser.add_argument("--metric_type", type=str, choices=["intrinsic", "extrinsic"], required=True)
    parser.add_argument("--calculate_metric_after_generation", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.metric_type == "intrinsic":
        assert args.api in ["hf"], "Intrinsic evaluation only supports hf API."

    args.model_name = args.model_ckpt.split("/")[-1]
    args.id_data_save_root = os.path.join(args.out_root, "id_data", args.model_name)
    args.ood_data_save_root = os.path.join(args.out_root, "ood_data", args.model_name)

    main(args)
