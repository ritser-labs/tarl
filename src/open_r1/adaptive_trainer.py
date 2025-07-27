# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import math
import numpy as np

from trl import GRPOTrainer
from transformers import PreTrainedTokenizer
from vllm import SamplingParams

logger = logging.getLogger(__name__)


class TokenAdaptiveGRPOTrainer(GRPOTrainer):
    """
    GRPO Trainer with token-adaptive reinforcement learning strategy.
    
    This trainer monitors uncertainty at each token during generation and truncates
    rollouts at failure points, then resamples from the truncated prefixes to focus
    computation on difficult reasoning segments.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Token-adaptive strategy parameters
        self.use_token_adaptive = getattr(self.args, 'use_token_adaptive', False)
        self.uncertainty_threshold = getattr(self.args, 'uncertainty_threshold', 0.7)
        self.entropy_threshold = getattr(self.args, 'entropy_threshold', 2.0)
        self.min_prefix_length = getattr(self.args, 'min_prefix_length', 10)
        self.max_resamples_per_prefix = getattr(self.args, 'max_resamples_per_prefix', 3)
        self.adaptive_temperature_scale = getattr(self.args, 'adaptive_temperature_scale', 1.2)
        
        # Statistics tracking
        self.adaptive_stats = {
            'total_rollouts': 0,
            'truncated_rollouts': 0,
            'resampled_rollouts': 0,
            'avg_failure_point': 0.0,
            'avg_uncertainty_at_failure': 0.0,
        }
        
        logger.info(f"TokenAdaptiveGRPOTrainer initialized with adaptive={self.use_token_adaptive}")
        if self.use_token_adaptive:
            logger.info(f"  - uncertainty_threshold: {self.uncertainty_threshold}")
            logger.info(f"  - entropy_threshold: {self.entropy_threshold}")
            logger.info(f"  - min_prefix_length: {self.min_prefix_length}")
            logger.info(f"  - max_resamples_per_prefix: {self.max_resamples_per_prefix}")

    def _detect_failure_point(self, logits: torch.Tensor, tokens: List[int], 
                             min_length: int = None) -> Optional[int]:
        """
        Detect the failure point in a sequence based on uncertainty metrics.
        
        Args:
            logits: Logits tensor of shape [seq_len, vocab_size]
            tokens: List of generated token IDs
            min_length: Minimum length before considering failure (overrides self.min_prefix_length)
            
        Returns:
            Token index where failure occurred, or None if no failure detected
        """
        if min_length is None:
            min_length = self.min_prefix_length
            
        if len(tokens) <= min_length:
            return None
            
        # Calculate probabilities and entropy for each position
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Look for failure points starting after min_length
        for i in range(min_length, len(tokens)):
            max_prob = max_probs[i].item()
            entropy = entropies[i].item()
            
            # Check if this token shows high uncertainty
            if max_prob < self.uncertainty_threshold or entropy > self.entropy_threshold:
                logger.debug(f"Failure point detected at token {i}: max_prob={max_prob:.3f}, entropy={entropy:.3f}")
                return i
                
        return None

    def _generate_with_uncertainty_monitoring(self, prompts: List[str], 
                                           sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        """
        Generate completions with token-level uncertainty monitoring.
        
        Returns list of completion dictionaries with uncertainty information.
        """
        if not self.use_token_adaptive:
            # Fall back to standard generation
            return self._standard_generation(prompts, sampling_params)
        
        all_completions = []
        
        for prompt in prompts:
            prompt_completions = []
            prefixes_to_resample = [(prompt, 0)]  # (prefix, resample_count)
            
            while prefixes_to_resample and len(prompt_completions) < self.args.num_generations:
                current_prefix, resample_count = prefixes_to_resample.pop(0)
                
                # Generate with logprobs to monitor uncertainty
                enhanced_params = SamplingParams(
                    temperature=sampling_params.temperature * (self.adaptive_temperature_scale ** resample_count),
                    top_p=sampling_params.top_p,
                    max_tokens=sampling_params.max_tokens,
                    logprobs=True,  # Enable logprobs for uncertainty monitoring
                    n=1,
                )
                
                try:
                    # Generate completion using vLLM
                    outputs = self.generation_model.generate([current_prefix], enhanced_params)
                    output = outputs[0]
                    
                    completion_text = output.outputs[0].text
                    token_ids = output.outputs[0].token_ids
                    logprobs_data = output.outputs[0].logprobs
                    
                    # Extract logits from logprobs (approximate)
                    logits = self._reconstruct_logits_from_logprobs(logprobs_data)
                    
                    # Detect failure point
                    failure_point = self._detect_failure_point(logits, token_ids)
                    
                    if failure_point is not None and resample_count < self.max_resamples_per_prefix:
                        # Truncate at failure point
                        truncated_tokens = token_ids[:failure_point]
                        truncated_text = self.processing_class.decode(truncated_tokens, skip_special_tokens=True)
                        truncated_prefix = current_prefix + truncated_text
                        
                        # Add truncated completion
                        completion_dict = {
                            'text': truncated_text,
                            'truncated': True,
                            'failure_point': failure_point,
                            'resample_count': resample_count,
                            'uncertainty_at_failure': self._get_uncertainty_at_position(logits, failure_point),
                        }
                        prompt_completions.append(completion_dict)
                        
                        # Schedule resampling from this prefix
                        prefixes_to_resample.append((truncated_prefix, resample_count + 1))
                        
                        # Update statistics
                        self.adaptive_stats['truncated_rollouts'] += 1
                        self.adaptive_stats['avg_failure_point'] += failure_point
                        
                        logger.debug(f"Truncated rollout at position {failure_point}, scheduling resample")
                        
                    else:
                        # Keep full completion
                        completion_dict = {
                            'text': completion_text,
                            'truncated': False,
                            'failure_point': None,
                            'resample_count': resample_count,
                            'uncertainty_at_failure': None,
                        }
                        prompt_completions.append(completion_dict)
                        
                        if resample_count > 0:
                            self.adaptive_stats['resampled_rollouts'] += 1
                    
                    self.adaptive_stats['total_rollouts'] += 1
                    
                except Exception as e:
                    logger.warning(f"Error in adaptive generation: {e}")
                    # Fall back to adding empty completion
                    completion_dict = {
                        'text': "",
                        'truncated': False,
                        'failure_point': None,
                        'resample_count': resample_count,
                        'uncertainty_at_failure': None,
                    }
                    prompt_completions.append(completion_dict)
                    
            all_completions.append(prompt_completions)
        
        return all_completions

    def _reconstruct_logits_from_logprobs(self, logprobs_data: List[Dict]) -> torch.Tensor:
        """
        Reconstruct approximate logits from vLLM logprobs output.
        
        This is an approximation since we don't have access to full logits,
        only the top-k logprobs from vLLM.
        """
        if not logprobs_data:
            return torch.zeros((0, self.processing_class.vocab_size))
            
        seq_len = len(logprobs_data)
        vocab_size = self.processing_class.vocab_size
        
        # Initialize with very negative values (representing low probability)
        logits = torch.full((seq_len, vocab_size), -10.0)
        
        for i, token_logprobs in enumerate(logprobs_data):
            if token_logprobs is None:
                continue
                
            # Set logprobs for tokens we have information about
            for token_id, logprob in token_logprobs.items():
                if isinstance(token_id, int) and 0 <= token_id < vocab_size:
                    logits[i, token_id] = logprob
        
        return logits

    def _get_uncertainty_at_position(self, logits: torch.Tensor, position: int) -> float:
        """Get uncertainty metric at a specific position."""
        if position >= logits.shape[0]:
            return 0.0
            
        probs = F.softmax(logits[position], dim=-1)
        max_prob = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        
        # Return combined uncertainty metric
        return entropy / max_prob

    def _standard_generation(self, prompts: List[str], sampling_params: SamplingParams) -> List[Dict[str, Any]]:
        """Standard generation without adaptive strategy."""
        outputs = self.generation_model.generate(prompts, sampling_params)
        
        completions = []
        for output in outputs:
            prompt_completions = []
            for generation in output.outputs:
                completion_dict = {
                    'text': generation.text,
                    'truncated': False,
                    'failure_point': None,
                    'resample_count': 0,
                    'uncertainty_at_failure': None,
                }
                prompt_completions.append(completion_dict)
            completions.append(prompt_completions)
            
        return completions

    def _generate_completions(self, prompts: List[str]) -> List[List[str]]:
        """
        Override the completion generation to use adaptive strategy.
        """
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=self.args.temperature,
            top_p=getattr(self.args, 'top_p', 1.0),
            max_tokens=self.args.max_completion_length,
            n=self.args.num_generations if not self.use_token_adaptive else 1,
        )
        
        # Generate with adaptive strategy
        adaptive_completions = self._generate_with_uncertainty_monitoring(prompts, sampling_params)
        
        # Convert back to the expected format for GRPO trainer
        standard_completions = []
        for prompt_completions in adaptive_completions:
            completions_list = [comp['text'] for comp in prompt_completions]
            # Pad to required number of generations if needed
            while len(completions_list) < self.args.num_generations:
                completions_list.append("")
            standard_completions.append(completions_list[:self.args.num_generations])
            
        return standard_completions

    def log_stats(self, logs: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Log adaptive strategy statistics."""
        logs = super().log_stats(logs, prefix)
        
        if self.use_token_adaptive and self.adaptive_stats['total_rollouts'] > 0:
            total_rollouts = self.adaptive_stats['total_rollouts']
            truncation_rate = self.adaptive_stats['truncated_rollouts'] / total_rollouts
            resample_rate = self.adaptive_stats['resampled_rollouts'] / total_rollouts
            
            avg_failure_point = (self.adaptive_stats['avg_failure_point'] / 
                               max(1, self.adaptive_stats['truncated_rollouts']))
            
            adaptive_logs = {
                f"{prefix}adaptive/truncation_rate": truncation_rate,
                f"{prefix}adaptive/resample_rate": resample_rate,
                f"{prefix}adaptive/avg_failure_point": avg_failure_point,
                f"{prefix}adaptive/total_rollouts": total_rollouts,
            }
            
            logs.update(adaptive_logs)
            
        return logs 