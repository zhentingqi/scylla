# Licensed under the MIT license.

from modules.Task import task2level
import json, os
from colorama import Fore, Back, Style

all_tasks = list(task2level.keys())
print(f"all tasks: {all_tasks}")
print(f"num tasks: {len(all_tasks)}")

DISABLE_WARNINGS = True

ignored_models = []


def check_file_empty(file_path):
    return os.stat(file_path).st_size == 0

def red(text):
    return f"{Fore.RED}{text}{Style.RESET_ALL}"

def yellow(text):
    return f"{Fore.YELLOW}{text}{Style.RESET_ALL}"

def green(text):
    return f"{Fore.GREEN}{text}{Style.RESET_ALL}"

def print_error(text):
    print(f"{red('ERROR')}: {text}")
    
def print_warning(text):
    if not DISABLE_WARNINGS:
        print(f"{yellow('WARNING')}: {text}")

src = "out"

#! Check `distrib_probing`
print(green("************* Checking `distrib_probing` *************"))
distrib_probing_dir = os.path.join(src, "distrib_probing")
for model_name in os.listdir(distrib_probing_dir):
    if model_name in ignored_models:
        continue
    
    print(f"==> Checking model {model_name}")
    model_dir = os.path.join(distrib_probing_dir, model_name)
    task_names = os.listdir(model_dir)
    
    for task_name in task_names:
        if task_name not in all_tasks:
            print_warning(f"task {task_name} is not a valid task")
    for task_name in all_tasks:
        if task_name not in task_names:
            print_error(f"task {task_name} is missing")
            
    for task_name in task_names:
        if task_name not in all_tasks:
            continue
        task_dir = os.path.join(model_dir, task_name)
        n_seeds = 0
        has_aggregation = False
        for f_name in os.listdir(task_dir):
            if not f_name.endswith(".json"):
                print_error(f"task {task_name} - File {f_name} is not a json file")
                
            if f_name == "all_outputs_across_seeds.json":
                if check_file_empty(os.path.join(task_dir, f_name)):
                    print_error(f"task {task_name} - {f_name} is empty")
                has_aggregation = True
                continue
            
            if not f_name.startswith("seed"):
                print_error(f"task {task_name} - File {f_name} is not a seed file")
                
            n_seeds += 1
        
        if not has_aggregation:
            print_error(f"task {task_name} - missing `all_outputs_across_seeds.json`")
            
        if n_seeds != 10:
            print_warning(f"task {task_name} - {n_seeds} seeds, usually expected 10")
            
#! Check `id_data`
print(green("************* Checking `id_data` *************"))
id_data_dir = os.path.join(src, "id_data")
if not os.path.exists(id_data_dir):
    print_error(f"`id_data` directory does not exist")
else:
    for model_name in os.listdir(id_data_dir):
        if model_name in ignored_models:
            continue
        
        print(f"==> Checking model {model_name}")
        model_dir = os.path.join(id_data_dir, model_name)
        task_names = os.listdir(model_dir)
        
        for task_name in task_names:
            if task_name not in all_tasks:
                print_warning(f"task {task_name} is not a valid task")
        for task_name in all_tasks:
            if task_name not in task_names:
                print_error(f"task {task_name} is missing")
                
        for task_name in task_names:
            if task_name not in all_tasks:
                continue
            task_dir = os.path.join(model_dir, task_name)
            wanted_files = ["all_id_elements.json", "all_id_eval_results---instruction---extrinsic.json", "all_id_examples.json", "all_id_io_pairs.json", "all_id_prompt_data---fewshot.json", "all_id_prompt_data---instruction.json", "all_id_prompt_data---mixed.json"]
            for f in wanted_files:
                if not os.path.exists(os.path.join(task_dir, f)):
                    print_error(f"task {task_name} - missing {f}")
                else:
                    if check_file_empty(os.path.join(task_dir, f)):
                        print_error(f"task {task_name} - {f} is empty")
            optional_files = ["all_id_eval_results---mixed---intrinsic.json", ]
            for f in optional_files:
                if not os.path.exists(os.path.join(task_dir, f)):
                    print_warning(f"task {task_name} - missing {f}")
                else:
                    if check_file_empty(os.path.join(task_dir, f)):
                        print_error(f"task {task_name} - {f} is empty")

#! Check `ood_data`
print(green("************* Checking `ood_data` *************"))
ood_data_dir = os.path.join(src, "ood_data")
if not os.path.exists(ood_data_dir):
    print_error(f"`ood_data` directory does not exist")
else:
    for model_name in os.listdir(ood_data_dir):
        if model_name in ignored_models:
            continue
        
        print(f"==> Checking model {model_name}")
        model_dir = os.path.join(ood_data_dir, model_name)
        task_names = os.listdir(model_dir)
        
        for task_name in task_names:
            if task_name not in all_tasks:
                print_warning(f"task {task_name} is not a valid task")
        for task_name in all_tasks:
            if task_name not in task_names:
                print_error(f"task {task_name} is missing")
                
        for task_name in task_names:
            if task_name not in all_tasks:
                continue
            task_dir = os.path.join(model_dir, task_name)
            wanted_files = ["all_ood_elements.json", "all_ood_eval_results---instruction---extrinsic.json", "all_ood_examples.json", "all_ood_io_pairs.json", "all_ood_prompt_data---fewshot.json", "all_ood_prompt_data---instruction.json", "all_ood_prompt_data---mixed.json"]
            for f in wanted_files:
                if not os.path.exists(os.path.join(task_dir, f)):
                    print_error(f"task {task_name} - missing {f}")
                else:
                    if check_file_empty(os.path.join(task_dir, f)):
                        print_error(f"task {task_name} - {f} is empty")
            optional_files = ["all_ood_eval_results---mixed---intrinsic.json", ]
            for f in optional_files:
                if not os.path.exists(os.path.join(task_dir, f)):
                    print_warning(f"task {task_name} - missing {f}")
                else:
                    if check_file_empty(os.path.join(task_dir, f)):
                        print_error(f"task {task_name} - {f} is empty")

#! Check `measurement`
print(green("************* Checking `measurement` *************"))
measurement_dir = os.path.join(src, "measurement")
if not os.path.exists(measurement_dir):
    print_error(f"`measurement` directory does not exist")
else:
    for model_name in os.listdir(measurement_dir):
        if model_name in ignored_models:
            continue
        
        print(f"==> Checking model {model_name}")
        model_dir = os.path.join(measurement_dir, model_name)
        task_names = os.listdir(model_dir)
        
        for task_name in task_names:
            if task_name not in all_tasks:
                print_warning(f"task {task_name} is not a valid task")
        for task_name in all_tasks:
            if task_name not in task_names:
                print_error(f"task {task_name} is missing")
                
        for task_name in task_names:
            if task_name not in all_tasks:
                continue
            task_dir = os.path.join(model_dir, task_name)
            wanted_files = ["measurement---instruction---extrinsic.json"]
            for f in wanted_files:
                if not os.path.exists(os.path.join(task_dir, f)):
                    print_error(f"task {task_name} - missing {f}")
                else:
                    if check_file_empty(os.path.join(task_dir, f)):
                        print_error(f"task {task_name} - {f} is empty")
            optional_files = ["all_id_normalized_results.json", "all_ood_normalized_results.json", "measurement---mixed---intrinsic.json"]
            for f in optional_files:
                if not os.path.exists(os.path.join(task_dir, f)):
                    print_warning(f"task {task_name} - missing {f}")
                else:
                    if check_file_empty(os.path.join(task_dir, f)):
                        print_error(f"task {task_name} - {f} is empty")
                        