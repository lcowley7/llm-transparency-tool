#!/bin/bash

# Evaluate two winogrande challenges with different checkpoints

# Initialize variables
revisions=(main $(seq -f "ckpt_%03g" 8 9 350) )
gpu_count=8

# Loop through the revisions in chunks of 8
for ((i=0; i<${#revisions[@]}; i+=gpu_count))
do
    for ((j=0; j<gpu_count; j++))
    do
        amber_rev=${revisions[i+j]}
        gpu=$j
        
        # Check if amber_rev is not empty (handles cases where revisions array is not a multiple of gpu_count)
        if [ -n "$amber_rev" ]; then
            # Execute the command in the background
            lm_eval --model hf --model_args pretrained=LLM360/amber,revision=$amber_rev --device cuda:$gpu --batch_size 8 --tasks winogrande --num_fewshot 5 --output_path ~/projects/harness-results/raw5shot/$amber_rev --log_samples &
            # lm_eval --model hf --model_args pretrained=LLM360/amber,revision=$amber_rev --device cuda:$gpu --batch_size 8 --tasks winogrande --num_fewshot 0 --output_path ~/projects/harness-results/raw0shot/$amber_rev --log_samples &
           
            # Optional: Output the command for debugging purposes
            echo "[AMBER EVAL ITER] Executing with amber_rev=$amber_rev on gpu=$gpu"
        fi
    done

    # Wait for all background jobs to finish before starting the next set
    wait
done