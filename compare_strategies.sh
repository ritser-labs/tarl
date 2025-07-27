#!/bin/bash

echo "==================================================="
echo "COMPARING BASELINE GRPO vs TOKEN-ADAPTIVE GRPO"
echo "==================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "openr1" ]; then
    echo "Error: Virtual environment 'openr1' not found. Please set up the environment first."
    exit 1
fi

# Function to run training and track time
run_training() {
    local strategy_name=$1
    local script_name=$2
    local config_name=$3
    
    echo "Starting $strategy_name training..."
    echo "Script: $script_name"
    echo "Config: $config_name"
    echo "Time: $(date)"
    echo ""
    
    start_time=$(date +%s)
    
    # Run the training
    if [ "$strategy_name" = "BASELINE" ]; then
        ./train_multi_gpu.sh --wandb_project "grpo-comparison" --hub_model_id "OpenRS-Baseline-$(date +%Y%m%d-%H%M%S)"
    else
        ./train_adaptive_grpo.sh --wandb_project "grpo-comparison" --hub_model_id "OpenRS-Adaptive-$(date +%Y%m%d-%H%M%S)"
    fi
    
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo ""
    echo "$strategy_name training completed!"
    echo "Duration: $((duration/3600))h $((duration%3600/60))m $((duration%60))s"
    echo "==================================================="
    echo ""
}

# Ask user which strategies to run
echo "Which training strategies would you like to run?"
echo "1. Baseline GRPO only"
echo "2. Token-Adaptive GRPO only"  
echo "3. Both (sequential)"
echo "4. Exit"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo "Running BASELINE GRPO training..."
        run_training "BASELINE" "train_multi_gpu.sh" "recipes/grpo.yaml"
        ;;
    2)
        echo "Running TOKEN-ADAPTIVE GRPO training..."
        run_training "TOKEN-ADAPTIVE" "train_adaptive_grpo.sh" "recipes/adaptive_grpo.yaml"
        ;;
    3)
        echo "Running BOTH strategies sequentially..."
        echo ""
        run_training "BASELINE" "train_multi_gpu.sh" "recipes/grpo.yaml"
        sleep 60  # Brief pause between trainings
        run_training "TOKEN-ADAPTIVE" "train_adaptive_grpo.sh" "recipes/adaptive_grpo.yaml"
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "==================================================="
echo "TRAINING COMPARISON COMPLETE!"
echo "==================================================="
echo ""
echo "Check your WandB project 'grpo-comparison' to see the results."
echo "Key metrics to compare:"
echo "- Training loss progression"
echo "- Reward scores (format, cosine)"
echo "- Sample efficiency"
echo "- Convergence speed"
echo ""
echo "For Token-Adaptive strategy, also check:"
echo "- adaptive/truncation_rate"
echo "- adaptive/resample_rate" 
echo "- adaptive/avg_failure_point"
echo "- adaptive/total_rollouts"
echo "" 