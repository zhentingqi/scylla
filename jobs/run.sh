#!/bin/bash

run_0() {
    local model_ckpt=$1
    local task_name=$2

    seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49) # as many seeds as possible to generate enough data 

    for seed in "${seeds[@]}"
    do

        echo "Running seed $seed"
        python 0_distrib_probing.py \
            --model_ckpt $model_ckpt \
            --task_name $task_name \
            --seed $seed \

    done

    python 0_5_aggregate_across_seeds.py \
        --model_ckpt $model_ckpt \
        --task_name $task_name \

}


run_1234() {
    local model_ckpt=$1
    local task_name=$2

    python 1_extract_id_data.py \
        --model_ckpt $model_ckpt \
        --task_name $task_name \

    python 2_gen_ood_data.py \
        --model_ckpt $model_ckpt \
        --task_name $task_name \

    python 3_get_io_pairs.py \
        --model_ckpt $model_ckpt \
        --task_name $task_name \

    python 4_make_prompt_data.py \
        --model_ckpt $model_ckpt \
        --task_name $task_name \

}


run_56() {
    local model_ckpt=$1
    local api=$2
    local task_name=$3
    local prompt_mode=$4
    local metric_type=$5

    python 5_eval_model.py \
        --model_ckpt $model_ckpt \
        --api $api \
        --task_name $task_name \
        --prompt_mode $prompt_mode \
        --metric_type $metric_type \

    python 5_5_process_eval_results.py \
        --model_ckpt $model_ckpt \
        --task_name $task_name \
        --prompt_mode $prompt_mode \
        --metric_type $metric_type \

    python 6_compute_measurement.py \
        --model_ckpt $model_ckpt \
        --task_name $task_name \
        --prompt_mode $prompt_mode \
        --metric_type $metric_type \
        
}

run_7() {
    local model_ckpt=$1
    local task_name=$2

    python 7_normalize_acc.py \
        --model_ckpt $model_ckpt \
        --task_name $task_name \

}


models=(meta-llama/Meta-Llama-3.1-8B-Instruct)
tasks=(FindMinimum FindMaximum FindMode FindTopk SortNumbers RemoveDuplicateNumbers LongestIncreasingSubsequence LongestConsecutiveElements TwoSum ThreeSum FourSum SubsetSum LongestCommonSubarray TSP ThreeSumInRange FourSumInRange SubsetSumInRange ThreeSumMultipleTen FourSumMultipleTen SubsetSumMultipleTen)

for m in "${models[@]}"
do
    for t in "${tasks[@]}"
    do

        #! model_ckpt=$1, task_name=$2
        run_0 $m $t

        #! model_ckpt=$1, task_name=$2
        run_1234 $m $t

        #! model_ckpt=$1, api=$2, task_name=$3, prompt_mode=$4, metric_type=$5
        run_56 $m vllm $t instruction extrinsic
        # run_56 $m hf $t mixed intrinsic   # (optional)

        #! model_ckpt=$1, task_name=$2
        # run_7 $m $t   # (optional)

    done
done
