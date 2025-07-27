#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, 'src')

import torch
import logging
from vllm import SamplingParams
from src.open_r1.adaptive_trainer import TokenAdaptiveGRPOTrainer
from transformers import AutoTokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_adaptive_logic():
    """Test the adaptive generation logic in isolation."""
    
    # Mock trainer args
    class MockArgs:
        use_token_adaptive = True
        uncertainty_threshold = 0.7
        entropy_threshold = 2.0
        min_prefix_length = 10
        max_resamples_per_prefix = 3
        adaptive_temperature_scale = 1.2
        temperature = 0.7
        max_completion_length = 100
        num_generations = 6
    
    # Create mock trainer
    trainer = TokenAdaptiveGRPOTrainer.__new__(TokenAdaptiveGRPOTrainer)
    trainer.args = MockArgs()
    trainer.use_token_adaptive = True
    trainer.uncertainty_threshold = 0.7
    trainer.entropy_threshold = 2.0
    trainer.min_prefix_length = 10
    trainer.max_resamples_per_prefix = 3
    trainer.adaptive_temperature_scale = 1.2
    trainer.adaptive_stats = {
        'total_rollouts': 0,
        'truncated_rollouts': 0,
        'resampled_rollouts': 0,
        'avg_failure_point': 0.0,
        'avg_uncertainty_at_failure': 0.0,
    }
    
    # Mock tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    trainer.processing_class = tokenizer
    
    # Mock generation model (we'll skip actual generation)
    class MockGenerationModel:
        def generate(self, prompts, params):
            # Mock output with some tokens
            class MockOutput:
                def __init__(self):
                    self.outputs = [MockGeneration()]
            
            class MockGeneration:
                def __init__(self):
                    self.text = "Let me think about this step by step. First, I need to..."
                    self.token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
                    # Mock logprobs with varying confidence
                    self.logprobs = [
                        {1: -0.1, 2: -2.0, 3: -3.0},  # High confidence
                        {2: -0.2, 1: -1.5, 4: -3.0},  # High confidence  
                        {3: -0.8, 2: -1.2, 1: -2.0},  # Medium confidence (might trigger)
                        {4: -0.5, 3: -1.0, 2: -2.5},  # Medium confidence
                        {5: -1.5, 4: -1.6, 6: -1.7},  # Low confidence (should trigger)
                        {6: -0.3, 5: -1.1, 7: -2.0},  # High confidence
                        {7: -0.4, 6: -1.3, 8: -2.1},  # High confidence
                        {8: -0.6, 7: -1.4, 9: -2.2},  # Medium confidence
                        {9: -0.2, 8: -1.5, 10: -2.3}, # High confidence
                        {10: -0.3, 9: -1.6, 11: -2.4}, # High confidence
                        {11: -0.4, 10: -1.7, 12: -2.5}, # High confidence
                        {12: -0.5, 11: -1.8, 13: -2.6}, # Medium confidence
                        {13: -0.6, 12: -1.9, 14: -2.7}, # Medium confidence
                        {14: -0.7, 13: -2.0, 15: -2.8}, # Medium confidence
                        {15: -0.8, 14: -2.1, 1: -2.9},  # Medium confidence
                    ]
            return [MockOutput()]
    
    trainer.generation_model = MockGenerationModel()
    
    # Test the method
    test_inputs = [
        {"prompt": "Solve this math problem: What is 2+2?"},
        {"prompt": "Explain photosynthesis in simple terms."},
    ]
    
    logger.info("=== TESTING ADAPTIVE LOGIC ===")
    
    try:
        result = trainer._generate_and_score_completions(test_inputs)
        logger.info(f"Test completed successfully!")
        return True
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_adaptive_logic()
    if success:
        print("✅ Adaptive logic test PASSED")
    else:
        print("❌ Adaptive logic test FAILED")
        sys.exit(1) 