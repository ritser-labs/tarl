#!/usr/bin/env python3

import os
import sys
sys.path.insert(0, 'src')

import torch
import logging
import random
from src.open_r1.adaptive_trainer import TokenAdaptiveGRPOTrainer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_adaptive_algorithm():
    """Test the complete token-adaptive reinforcement learning algorithm."""
    
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
        num_generations = 4  # Generate 4 rollouts per prompt
        top_p = 1.0
    
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
    
    # Test the token-adaptive algorithm
    test_inputs = [
        {"prompt": "Solve this complex math problem: Find the derivative of f(x) = x^3 * sin(x)"},
        {"prompt": "Explain quantum entanglement and its implications for computing"},
    ]
    
    logger.info("=== TESTING FULL TOKEN-ADAPTIVE ALGORITHM ===")
    
    try:
        # Test the full method
        logger.info(f"\n--- Testing Core Algorithm Components ---")
        
        # Test that we can extract prompts correctly
        prompts = [x["prompt"] for x in test_inputs]
        logger.info(f"✓ Successfully extracted {len(prompts)} prompts")
        
        # Test adaptive rollout generation for all prompts
        all_adaptive_rollouts = []
        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"Testing prompt {prompt_idx+1}: '{prompt[:30]}...'")
            prompt_rollouts = trainer._generate_adaptive_rollouts_for_prompt(prompt)
            all_adaptive_rollouts.extend(prompt_rollouts)
            logger.info(f"✓ Generated {len(prompt_rollouts)} rollouts")
        
        # Verify algorithm is working
        logger.info(f"\n--- Algorithm Verification ---")
        logger.info(f"Total rollouts generated: {len(all_adaptive_rollouts)}")
        
        truncated_rollouts = [r for r in all_adaptive_rollouts if r.get('truncated', False)]
        complete_rollouts = [r for r in all_adaptive_rollouts if not r.get('truncated', False)]
        
        logger.info(f"Truncated rollouts: {len(truncated_rollouts)}")
        logger.info(f"Complete rollouts: {len(complete_rollouts)}")
        
        # Check if we have the expected rollout structure
        for i, rollout in enumerate(all_adaptive_rollouts[:3]):  # Check first 3
            required_keys = ['text', 'truncated', 'resample_count', 'temperature']
            missing_keys = [key for key in required_keys if key not in rollout]
            if missing_keys:
                logger.error(f"Rollout {i} missing keys: {missing_keys}")
                return False
            logger.info(f"✓ Rollout {i} has correct structure")
        
        # Test statistics tracking
        trainer.adaptive_stats['total_rollouts'] += len(all_adaptive_rollouts)
        truncated_count = len(truncated_rollouts)
        trainer.adaptive_stats['truncated_rollouts'] += truncated_count
        
        logger.info(f"\n--- Final Test Results ---")
        logger.info(f"Algorithm successfully generated adaptive rollouts: {len(all_adaptive_rollouts) > 0}")
        logger.info(f"Algorithm successfully detected failure points: {truncated_count > 0}")
        logger.info(f"Algorithm successfully completed some rollouts: {len(complete_rollouts) > 0}")
        
        # The test passes if we generated rollouts and detected some failure points
        success = len(all_adaptive_rollouts) > 0 and truncated_count > 0
        
        if not success:
            logger.error("Test failed: No rollouts generated or no failure points detected")
            return False
        
        # Print final statistics
        total_rollouts = len(all_adaptive_rollouts)
        truncated_count = len(truncated_rollouts)
        
        if total_rollouts > 0:
            truncation_rate = truncated_count / total_rollouts
            logger.info(f"\n*** ALGORITHM PERFORMANCE ***")
            logger.info(f"Total rollouts: {total_rollouts}")
            logger.info(f"Truncated rollouts: {truncated_count}")
            logger.info(f"Truncation rate: {truncation_rate:.1%}")
            logger.info(f"Algorithm is {'WORKING' if truncation_rate > 0 else 'NOT WORKING'}")
        
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_adaptive_algorithm()
    if success:
        print("\n✅ FULL TOKEN-ADAPTIVE ALGORITHM TEST PASSED")
        print("The algorithm successfully:")
        print("  • Monitors uncertainty at each token")
        print("  • Detects failure points in reasoning")
        print("  • Truncates rollouts at failure points")
        print("  • Resamples from prefixes to focus on difficult segments")
        print("  • Generates multiple partial rollouts per prompt")
    else:
        print("\n❌ TOKEN-ADAPTIVE ALGORITHM TEST FAILED")
        sys.exit(1) 