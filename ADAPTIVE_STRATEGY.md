# Token-Adaptive Reinforcement Learning Strategy

## Overview

This implementation introduces a novel **token-adaptive reinforcement learning strategy** for enhancing small language models' reasoning capabilities. Instead of treating all generated tokens uniformly, this approach monitors the model's uncertainty at each token position during generation and dynamically adapts the training process.

## Key Innovation

### The Problem
Traditional RL approaches for LLMs generate complete sequences and treat them uniformly during training. This wastes computation on tokens the model already handles well while under-exploring difficult reasoning segments where the model struggles.

### The Solution
Our token-adaptive strategy:

1. **Monitors uncertainty in real-time** during generation using logit entropy and max probability
2. **Detects failure points** where the model becomes uncertain or starts to diverge
3. **Truncates rollouts** at these failure points instead of continuing with likely poor completions
4. **Resamples from prefixes** to generate multiple attempts at the challenging segments
5. **Focuses compute** on the most confusing parts of the reasoning trajectory

## Implementation Details

### Core Components

1. **TokenAdaptiveGRPOTrainer** (`src/open_r1/adaptive_trainer.py`)
   - Extends the standard GRPOTrainer
   - Implements uncertainty monitoring during vLLM generation
   - Handles rollout truncation and prefix resampling logic

2. **Configuration Parameters** (`src/open_r1/configs.py`)
   - `use_token_adaptive`: Enable/disable the adaptive strategy
   - `uncertainty_threshold`: Max probability threshold for failure detection (default: 0.7)
   - `entropy_threshold`: Entropy threshold for failure detection (default: 2.0)
   - `min_prefix_length`: Minimum tokens before considering truncation (default: 15)
   - `max_resamples_per_prefix`: Maximum resamples from same prefix (default: 3)
   - `adaptive_temperature_scale`: Temperature scaling for resampling (default: 1.2)

### Uncertainty Detection Algorithm

For each generated token at position `i`:
```python
# Calculate probability distribution
probs = softmax(logits[i])
max_prob = max(probs)
entropy = -sum(p * log(p) for p in probs)

# Detect failure point if:
if max_prob < uncertainty_threshold or entropy > entropy_threshold:
    failure_point = i
    truncate_and_resample()
```

### Adaptive Sampling Process

1. **Generate with logprobs enabled** to monitor uncertainty
2. **Scan each token** for failure indicators
3. **When failure detected**:
   - Truncate completion at failure point
   - Add truncated completion to training data
   - Schedule resampling from the truncated prefix
   - Increase temperature for subsequent attempts
4. **Continue until** reaching target number of generations or max resamples

## Usage

### Basic Training

Run the token-adaptive strategy:
```bash
./train_adaptive_grpo.sh
```

### Comparison with Baseline

Compare both strategies:
```bash
./compare_strategies.sh
```

### Configuration

Modify `recipes/adaptive_grpo.yaml` to adjust parameters:
```yaml
# Token-Adaptive RL Strategy Parameters
use_token_adaptive: true
uncertainty_threshold: 0.7  # Lower = more sensitive to uncertainty
entropy_threshold: 2.0      # Higher = less sensitive to entropy
min_prefix_length: 15       # Minimum reasoning before truncation
max_resamples_per_prefix: 3 # How many times to retry difficult segments
adaptive_temperature_scale: 1.2  # Temperature increase per retry
```

## Expected Benefits

### Sample Efficiency
- **Focused exploration** on difficult reasoning segments
- **Reduced waste** on well-handled token sequences
- **Multiple attempts** at challenging parts instead of single long rollouts

### Training Stability
- **Early truncation** prevents cascading errors in long sequences
- **Adaptive temperature** provides more diversity when needed
- **Prefix reuse** leverages good reasoning starts multiple times

### Performance Gains
- **Better reasoning accuracy** through focused practice on hard segments
- **Improved sample efficiency** with fewer total tokens needed
- **Cost reduction** by avoiding unnecessary long completions

## Monitoring and Metrics

The adaptive strategy tracks several key metrics in WandB:

- `adaptive/truncation_rate`: Percentage of rollouts truncated due to uncertainty
- `adaptive/resample_rate`: Percentage of rollouts that are resamples
- `adaptive/avg_failure_point`: Average token position where failures occur
- `adaptive/total_rollouts`: Total number of rollouts processed

## Files Created

### Core Implementation
- `src/open_r1/adaptive_trainer.py` - Main trainer class
- `src/open_r1/adaptive_grpo.py` - Training script
- `src/open_r1/configs.py` - Updated with adaptive parameters

### Configuration and Scripts
- `recipes/adaptive_grpo.yaml` - Adaptive strategy configuration
- `train_adaptive_grpo.sh` - Training execution script
- `compare_strategies.sh` - Comparison utility
- `ADAPTIVE_STRATEGY.md` - This documentation

## Research Comparison

This approach enables direct comparison with the baseline GRPO method on:
- **Mathematical reasoning tasks** (AMC, AIME, MATH)
- **Training efficiency** (cost per performance point)
- **Sample efficiency** (tokens needed for convergence)
- **Reasoning quality** (step-by-step accuracy)

Run both strategies and analyze the WandB logs to quantify the improvements from token-adaptive RL.

## Next Steps

1. **Run the comparison** using `./compare_strategies.sh`
2. **Analyze WandB metrics** to quantify improvements
3. **Tune hyperparameters** based on initial results
4. **Evaluate on reasoning benchmarks** to measure final performance
5. **Compare costs** and sample efficiency gains

The token-adaptive strategy represents a significant step toward more intelligent and efficient reinforcement learning for language model reasoning. 