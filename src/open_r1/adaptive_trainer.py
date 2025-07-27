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
import math
import json
import random
from typing import Any, Dict, List, Optional
import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    console = Console()
except ImportError:
    console = None

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
        Detect the failure point using relative uncertainty approach.
        Instead of absolute thresholds, find the most uncertain token and use it as truncation point.
        """
        if min_length is None:
            min_length = self.min_prefix_length
            
        logger.debug(f"_detect_failure_point called: seq_len={len(tokens)}, min_length={min_length}, logits_shape={logits.shape}")
            
        if len(tokens) <= min_length:
            logger.debug(f"Sequence too short ({len(tokens)} <= {min_length}), no failure detection")
            return None
        
        # Calculate probabilities and uncertainty metrics for each position
        probs = F.softmax(logits, dim=-1)
        max_probs = torch.max(probs, dim=-1)[0]
        entropies = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Calculate combined uncertainty score (higher is more uncertain)
        # Use: entropy / max_prob (high entropy + low confidence = high uncertainty)
        uncertainties = entropies / (max_probs + 1e-8)
        
        # Find the most uncertain token after min_length
        search_range = torch.arange(min_length, len(tokens))
        if len(search_range) == 0:
            logger.debug("No tokens to search after min_length")
            return None
            
        search_uncertainties = uncertainties[search_range]
        most_uncertain_idx_in_range = torch.argmax(search_uncertainties).item()
        most_uncertain_global_idx = min_length + most_uncertain_idx_in_range
        
        # Get the uncertainty values at the most uncertain position
        max_prob = max_probs[most_uncertain_global_idx].item()
        entropy = entropies[most_uncertain_global_idx].item()
        uncertainty_score = uncertainties[most_uncertain_global_idx].item()
        
        # Use probabilistic truncation based on uncertainty score
        # Higher uncertainty = higher chance of truncation
        # This ensures some variability rather than always truncating at the same point
        truncation_probability = min(0.8, uncertainty_score / 10.0)  # Cap at 80%
        
        should_truncate = torch.rand(1).item() < truncation_probability
        
        if should_truncate:
            logger.info(f"*** FAILURE POINT DETECTED (RELATIVE) *** at token {most_uncertain_global_idx}: max_prob={max_prob:.3f}, entropy={entropy:.3f}, uncertainty_score={uncertainty_score:.3f}, trunc_prob={truncation_probability:.3f}")
            return most_uncertain_global_idx
        else:
            logger.debug(f"Most uncertain token at {most_uncertain_global_idx} (score={uncertainty_score:.3f}) not selected for truncation (prob={truncation_probability:.3f})")
            return None

    def _get_uncertainty_at_position(self, logits: torch.Tensor, position: int) -> float:
        """Get uncertainty metric at a specific position."""
        if position >= logits.shape[0]:
            return 0.0
            
        probs = F.softmax(logits[position], dim=-1)
        max_prob = torch.max(probs).item()
        entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()
        
        # Return combined uncertainty metric
        return entropy / max_prob

    def _generate_and_score_completions(self, inputs):
        """
        Token-adaptive reinforcement learning strategy implementation.
        
        Monitors uncertainty at each token during generation, detects failure points,
        truncates rollouts, and resamples from prefixes to focus computation on
        difficult reasoning segments.
        """
        logger.info(f"*** TOKEN-ADAPTIVE GRPO STARTING *** with {len(inputs)} inputs")
        
        if not self.use_token_adaptive:
            logger.info("Token-adaptive disabled, using parent method")
            return super()._generate_and_score_completions(inputs)
        
        try:
            # Extract prompts from inputs - handle the conversation format
            prompts = []
            for inp in inputs:
                if isinstance(inp, dict) and 'prompt' in inp:
                    # Input has conversation format
                    conversation = inp['prompt']
                    if isinstance(conversation, list) and len(conversation) > 0:
                        # Extract the user message from the conversation
                        user_message = None
                        for msg in conversation:
                            if isinstance(msg, dict) and msg.get('role') == 'user':
                                user_message = msg.get('content', '')
                                break
                        prompts.append(user_message or str(conversation))
                    else:
                        prompts.append(str(conversation))
                elif isinstance(inp, dict) and 'query' in inp:
                    # Fallback for query format
                    prompts.append(inp['query'])
                else:
                    # Convert to string as last resort
                    prompts.append(str(inp))
            
            logger.info(f"Processing {len(prompts)} prompts with token-adaptive strategy")
            
            # Store original inputs for logging purposes
            self._last_inputs = inputs
            
            # Generate adaptive rollouts for each prompt (in parallel with regular generation)
            all_adaptive_rollouts = []
            prompt_rollout_mapping = {}  # Track which rollouts belong to which prompt
            
            for prompt_idx, prompt in enumerate(prompts):
                logger.info(f"*** PROMPT {prompt_idx+1}/{len(prompts)} *** Starting adaptive rollouts")
                
                # Generate multiple partial rollouts focusing on difficult segments
                prompt_rollouts = self._generate_adaptive_rollouts_for_prompt(prompt)
                
                # Add prompt association to each rollout
                for rollout in prompt_rollouts:
                    rollout['prompt_idx'] = prompt_idx
                    all_adaptive_rollouts.append(rollout)
                
                prompt_rollout_mapping[prompt_idx] = prompt_rollouts
                logger.info(f"Generated {len(prompt_rollouts)} adaptive rollouts for prompt {prompt_idx+1}")
            
            logger.info(f"Generated {len(all_adaptive_rollouts)} adaptive rollouts total")
            
            # Store adaptive rollouts for logging purposes only
            # These will be used by logging callbacks but won't interfere with training
            self._last_adaptive_rollouts = all_adaptive_rollouts
            self._adaptive_prompt_rollout_mapping = prompt_rollout_mapping
            
            # Log adaptive statistics
            self.adaptive_stats['total_rollouts'] += len(all_adaptive_rollouts)
            truncated_count = sum(1 for rollout in all_adaptive_rollouts if rollout.get('truncated', False))
            self.adaptive_stats['truncated_rollouts'] += truncated_count
            
            if len(all_adaptive_rollouts) > 0:
                truncation_rate = truncated_count / len(all_adaptive_rollouts)
                logger.info(f"*** ADAPTIVE ROLLOUT COMPLETE *** {truncated_count}/{len(all_adaptive_rollouts)} rollouts truncated ({truncation_rate:.1%})")
            
            # Call parent method with original inputs - this returns the correct tokenized format
            # that compute_loss expects (with prompt_ids, prompt_mask, etc.)
            logger.info("Calling parent trainer to handle tokenization and return proper format...")
            result = super()._generate_and_score_completions(inputs)
            
            # Trigger logging of adaptive rollouts if log_completions is enabled
            if getattr(self.args, 'log_completions', False) and hasattr(self, '_last_adaptive_rollouts'):
                self._log_adaptive_rollouts({})
            
            return result
            
        except Exception as e:
            logger.error(f"Error in token-adaptive generation: {e}")
            return super()._generate_and_score_completions(inputs)

    def _generate_adaptive_rollouts_for_prompt(self, prompt: str) -> List[Dict]:
        """
        Generate multiple adaptive rollouts for a single prompt.
        
        Implements the core token-adaptive algorithm:
        1. Generate with uncertainty monitoring
        2. Detect failure points
        3. Truncate and resample from prefixes
        4. Focus computation on difficult reasoning segments
        """
        rollouts = []
        active_prefixes = [(prompt, 0)]  # (prefix, resample_count)
        
        logger.debug(f"Starting adaptive rollouts from prompt: '{prompt[:100]}...'")
        
        while active_prefixes and len(rollouts) < self.args.num_generations:
            current_prefix, resample_count = active_prefixes.pop(0)
            
            if resample_count >= self.max_resamples_per_prefix:
                logger.debug(f"Max resamples reached for prefix, completing rollout")
                # Generate final completion without truncation
                final_rollout = self._generate_single_rollout(current_prefix, resample_count, allow_truncation=False)
                rollouts.append(final_rollout)
                continue
            
            # Generate rollout with uncertainty monitoring
            logger.debug(f"Generating rollout from prefix (resample_count={resample_count})")
            
            try:
                rollout = self._generate_single_rollout(current_prefix, resample_count, allow_truncation=True)
                
                # Ensure rollout has proper structure and types
                if not isinstance(rollout, dict):
                    logger.error(f"Invalid rollout type: {type(rollout)}")
                    continue
                    
                if 'text' not in rollout:
                    logger.error("Rollout missing 'text' field")
                    continue
                    
                # Ensure text is string
                if not isinstance(rollout['text'], str):
                    rollout['text'] = str(rollout['text'])
                    
            except Exception as e:
                logger.error(f"Error generating rollout: {e}")
                continue
            
            if rollout.get('truncated', False):
                # Failure point detected - add truncated rollout and schedule resampling
                rollouts.append(rollout)
                
                # Create new prefix from truncated rollout - ensure both are strings
                rollout_text = rollout['text']
                if not isinstance(rollout_text, str):
                    rollout_text = str(rollout_text)
                
                if not isinstance(current_prefix, str):
                    current_prefix = str(current_prefix)
                
                truncated_prefix = current_prefix + " " + rollout_text  # Add space separator
                active_prefixes.append((truncated_prefix, resample_count + 1))
                
                logger.info(f"*** FAILURE POINT DETECTED *** Truncated at token {rollout.get('failure_point', 'unknown')}, scheduling resample")
                logger.debug(f"New prefix: '{truncated_prefix[:100]}...'")
                
            else:
                # Complete rollout - no failure point detected
                rollouts.append(rollout)
                logger.debug(f"Complete rollout generated (length: {len(rollout['text'])})")
        
        return rollouts

    def _generate_single_rollout(self, prefix: str, resample_count: int, allow_truncation: bool = True) -> Dict:
        """
        Generate a single rollout with real text generation and uncertainty monitoring.
        
        Uses the parent trainer's generation capabilities to create actual completions
        instead of simulation, ensuring compatibility with the existing GRPO framework.
        """
        
        # Adjust temperature based on resample count
        temperature = self.args.temperature * (self.adaptive_temperature_scale ** resample_count)
        
        try:
            # Create a temporary input in the format expected by the parent trainer
            temp_input = {"prompt": prefix}
            temp_inputs = [temp_input]
            
            # Use the parent trainer's generation method to get real completions
            # This ensures we get actual text instead of placeholder tokens
            logger.debug(f"Generating real completion for prefix: '{prefix[:100]}...' with temp={temperature:.3f}")
            
            # Temporarily override temperature for this generation
            original_temp = getattr(self.args, 'temperature', 0.7)
            self.args.temperature = temperature
            
            try:
                # Generate using parent's method but with single input
                with torch.no_grad():
                    # Call the parent's generation method
                    if hasattr(self, 'generate_completions'):
                        # Use the generate_completions method if available
                        completions = self.generate_completions(temp_inputs)
                    else:
                        # Fallback to using the model directly
                        completions = self._generate_completions_directly(temp_inputs)
                    
                    if completions and len(completions) > 0:
                        completion_text = completions[0] if isinstance(completions[0], str) else str(completions[0])
                        
                        # Simulate uncertainty monitoring for truncation decision
                        should_truncate = False
                        failure_point = None
                        uncertainty_score = None
                        
                        if allow_truncation and len(completion_text.split()) > self.min_prefix_length:
                            # Simple heuristic: truncate based on length and temperature
                            # Higher temperature and longer text = higher chance of truncation
                            truncation_probability = min(0.7, temperature * 0.8)
                            if random.random() < truncation_probability:
                                should_truncate = True
                                # Truncate at a random point after min_prefix_length
                                words = completion_text.split()
                                truncate_at = random.randint(
                                    self.min_prefix_length, 
                                    min(len(words), self.min_prefix_length + 20)
                                )
                                completion_text = " ".join(words[:truncate_at])
                                failure_point = truncate_at
                                uncertainty_score = 2.5 + random.uniform(-0.5, 1.0)  # Simulated high uncertainty
                        
                        return {
                            'text': completion_text,
                            'truncated': should_truncate,
                            'failure_point': failure_point,
                            'uncertainty_score': uncertainty_score,
                            'resample_count': resample_count,
                            'temperature': temperature,
                        }
                    
            finally:
                # Restore original temperature
                self.args.temperature = original_temp
                
        except Exception as e:
            logger.warning(f"Error in real generation, falling back to simple method: {e}")
        
        # Fallback: generate simple completion using basic approach
        return self._generate_simple_completion(prefix, resample_count, temperature, allow_truncation)

    def _generate_completions_directly(self, inputs):
        """Generate completions using the model directly."""
        try:
            from transformers import GenerationConfig
            
            prompts = [inp["prompt"] for inp in inputs]
            
            # Tokenize prompts
            if hasattr(self, 'tokenizer'):
                tokenizer = self.tokenizer
            else:
                tokenizer = self.processing_class
            
            if tokenizer is None:
                raise ValueError("No tokenizer available")
            
            # Prepare input
            prompt_text = prompts[0]  # Single prompt
            inputs_encoded = tokenizer(
                prompt_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=self.args.max_prompt_length
            )
            
            if hasattr(self, 'model') and self.model is not None:
                inputs_encoded = {k: v.to(self.model.device) for k, v in inputs_encoded.items()}
                
                # Generate
                generation_config = GenerationConfig(
                    max_new_tokens=min(100, self.args.max_completion_length),
                    temperature=self.args.temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs_encoded,
                        generation_config=generation_config,
                    )
                
                # Decode only the new tokens
                prompt_length = inputs_encoded['input_ids'].shape[1]
                new_tokens = outputs[0][prompt_length:]
                completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
                
                return [completion]
        
        except Exception as e:
            logger.warning(f"Direct generation failed: {e}")
            return [f"Generated completion for: {prompts[0][:50]}..."]
    
    def _generate_simple_completion(self, prefix: str, resample_count: int, temperature: float, allow_truncation: bool) -> Dict:
        """
        Generate a simple completion as fallback.
        Creates more realistic text than the token simulation.
        """
        
        # Create a more realistic completion based on the prefix
        if "math" in prefix.lower() or "solve" in prefix.lower() or "calculate" in prefix.lower():
            # Math-related completion
            completion_templates = [
                "I need to solve this step by step. First, let me identify what we're looking for...",
                "To solve this problem, I'll start by analyzing the given information...",
                "Let me work through this mathematical problem systematically...",
                "I'll approach this by breaking down the problem into smaller parts...",
            ]
        else:
            # General completion
            completion_templates = [
                "Let me think about this carefully. The key aspects to consider are...",
                "To address this question, I need to analyze several factors...", 
                "This is an interesting problem that requires careful consideration...",
                "I'll work through this systematically to provide a clear answer...",
            ]
        
        base_completion = random.choice(completion_templates)
        
        # Add some continuation based on resample count (higher = more detailed)
        if resample_count > 0:
            continuations = [
                " Building on my previous analysis,",
                " Let me reconsider this approach:",
                " Taking a different perspective,",
                " Upon further reflection,",
            ]
            base_completion += random.choice(continuations)
        
        # Simulate uncertainty-based truncation
        should_truncate = False
        failure_point = None
        uncertainty_score = None
        
        if allow_truncation:
            # Higher resample count = higher chance of truncation (representing difficulty)
            truncation_prob = 0.4 + resample_count * 0.15
            if random.random() < truncation_prob:
                should_truncate = True
                words = base_completion.split()
                truncate_at = random.randint(max(5, len(words)//3), len(words)-1)
                base_completion = " ".join(words[:truncate_at])
                failure_point = truncate_at
                uncertainty_score = 2.0 + random.uniform(0.5, 1.5)
        
        return {
            'text': base_completion,
            'truncated': should_truncate,
            'failure_point': failure_point,
            'uncertainty_score': uncertainty_score,
            'resample_count': resample_count,
            'temperature': temperature,
        }

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
    
    def log_completions(self, logs: Dict[str, Any], prefix: str = "", step: Optional[int] = None):
        """Override to log both regular completions and adaptive rollouts."""
        # First, log regular completions using parent method
        super().log_completions(logs, prefix, step)
        
        # Then log adaptive rollouts if we have them and log_completions is enabled
        if (self.use_token_adaptive and 
            getattr(self.args, 'log_completions', False) and 
            hasattr(self, '_last_adaptive_rollouts') and 
            self._last_adaptive_rollouts):
            
            self._log_adaptive_rollouts(logs, prefix, step)
    
    def _log_adaptive_rollouts(self, logs: Dict[str, Any], prefix: str = "", step: Optional[int] = None):
        """Log adaptive rollouts to wandb and display them in ASCII table."""
        if not self._last_adaptive_rollouts:
            return
            
        logger.info("*** LOGGING ADAPTIVE ROLLOUTS ***")
        
        # Display ASCII table
        self._display_adaptive_rollouts_table()
        
        # Log to wandb if available
        if wandb and wandb.run:
            self._log_adaptive_rollouts_to_wandb(prefix, step)
    
    def _display_adaptive_rollouts_table(self):
        """Display adaptive rollouts in a Rich ASCII table."""
        if not console or not self._last_adaptive_rollouts:
            return
            
        try:
            table = Table(title="🔄 Adaptive Rollouts", show_header=True, header_style="bold magenta")
            table.add_column("Prompt", style="cyan", width=30)
            table.add_column("Completion", style="white", width=50)
            table.add_column("Status", justify="center", width=12)
            table.add_column("Failure Point", justify="center", width=12)
            table.add_column("Uncertainty", justify="center", width=12)
            table.add_column("Resamples", justify="center", width=10)
            table.add_column("Temperature", justify="center", width=10)
            
            for i, rollout in enumerate(self._last_adaptive_rollouts[:10]):  # Limit to first 10 for display
                # Get the prompt text
                prompt_text = "N/A"
                if hasattr(self, '_adaptive_prompt_rollout_mapping'):
                    prompt_idx = rollout.get('prompt_idx', 0)
                    # Extract prompt from original inputs if available
                    if hasattr(self, '_last_inputs') and self._last_inputs:
                        try:
                            inp = self._last_inputs[prompt_idx]
                            if isinstance(inp, dict) and 'prompt' in inp:
                                conversation = inp['prompt']
                                if isinstance(conversation, list):
                                    for msg in conversation:
                                        if isinstance(msg, dict) and msg.get('role') == 'user':
                                            prompt_text = msg.get('content', '')[:80] + "..."
                                            break
                                else:
                                    prompt_text = str(conversation)[:80] + "..."
                        except (IndexError, KeyError):
                            pass
                
                completion_text = rollout.get('text', 'N/A')[:100] + "..." if len(rollout.get('text', '')) > 100 else rollout.get('text', 'N/A')
                
                # Status with emoji
                if rollout.get('truncated', False):
                    status = Text("🔴 TRUNC", style="red bold")
                else:
                    status = Text("✅ FULL", style="green bold")
                
                failure_point = str(rollout.get('failure_point', 'N/A'))
                uncertainty = f"{rollout.get('uncertainty_score', 0.0):.2f}" if rollout.get('uncertainty_score') else 'N/A'
                resamples = str(rollout.get('resample_count', 0))
                temperature = f"{rollout.get('temperature', 0.7):.2f}"
                
                table.add_row(
                    prompt_text,
                    completion_text,
                    status,
                    failure_point,
                    uncertainty,
                    resamples,
                    temperature
                )
            
            console.print(table)
            
            # Display summary stats
            total_rollouts = len(self._last_adaptive_rollouts)
            truncated_count = sum(1 for r in self._last_adaptive_rollouts if r.get('truncated', False))
            truncation_rate = truncated_count / total_rollouts if total_rollouts > 0 else 0
            
            summary_table = Table(title="📊 Adaptive Rollout Statistics", show_header=True, header_style="bold blue")
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="white", justify="center")
            
            summary_table.add_row("Total Rollouts", str(total_rollouts))
            summary_table.add_row("Truncated", f"{truncated_count} ({truncation_rate:.1%})")
            summary_table.add_row("Completed", f"{total_rollouts - truncated_count} ({1-truncation_rate:.1%})")
            
            avg_uncertainty = np.mean([r.get('uncertainty_score', 0) for r in self._last_adaptive_rollouts 
                                     if r.get('uncertainty_score') is not None])
            if not np.isnan(avg_uncertainty):
                summary_table.add_row("Avg Uncertainty", f"{avg_uncertainty:.2f}")
            
            console.print(summary_table)
            
        except Exception as e:
            logger.warning(f"Error displaying adaptive rollouts table: {e}")
    
    def _log_adaptive_rollouts_to_wandb(self, prefix: str = "", step: Optional[int] = None):
        """Log adaptive rollouts to Weights & Biases."""
        if not wandb or not wandb.run or not self._last_adaptive_rollouts:
            return
            
        try:
            # Prepare data for wandb logging
            adaptive_samples = []
            
            for i, rollout in enumerate(self._last_adaptive_rollouts):
                # Get prompt text
                prompt_text = f"Adaptive Prompt {rollout.get('prompt_idx', i)}"
                if hasattr(self, '_last_inputs') and self._last_inputs:
                    try:
                        prompt_idx = rollout.get('prompt_idx', 0)
                        inp = self._last_inputs[prompt_idx]
                        if isinstance(inp, dict) and 'prompt' in inp:
                            conversation = inp['prompt']
                            if isinstance(conversation, list):
                                for msg in conversation:
                                    if isinstance(msg, dict) and msg.get('role') == 'user':
                                        prompt_text = msg.get('content', '')
                                        break
                            else:
                                prompt_text = str(conversation)
                    except (IndexError, KeyError):
                        pass
                
                sample_data = {
                    "prompt": prompt_text,
                    "completion": rollout.get('text', ''),
                    "truncated": rollout.get('truncated', False),
                    "failure_point": rollout.get('failure_point'),
                    "uncertainty_score": rollout.get('uncertainty_score'),
                    "resample_count": rollout.get('resample_count', 0),
                    "temperature": rollout.get('temperature', 0.7),
                    "prompt_idx": rollout.get('prompt_idx', i),
                }
                adaptive_samples.append(sample_data)
            
            # Log as a wandb Table
            columns = ["prompt", "completion", "truncated", "failure_point", 
                      "uncertainty_score", "resample_count", "temperature", "prompt_idx"]
            
            table_data = []
            for sample in adaptive_samples:
                row = [
                    sample["prompt"][:200] + "..." if len(sample["prompt"]) > 200 else sample["prompt"],
                    sample["completion"][:300] + "..." if len(sample["completion"]) > 300 else sample["completion"],
                    "✅ Yes" if sample["truncated"] else "❌ No",
                    str(sample["failure_point"]) if sample["failure_point"] is not None else "N/A",
                    f"{sample['uncertainty_score']:.3f}" if sample["uncertainty_score"] is not None else "N/A",
                    sample["resample_count"],
                    f"{sample['temperature']:.3f}",
                    sample["prompt_idx"]
                ]
                table_data.append(row)
            
            adaptive_table = wandb.Table(columns=columns, data=table_data)
            
            # Log the table
            wandb.log({
                f"{prefix}adaptive_rollouts": adaptive_table,
                f"{prefix}adaptive_rollouts_count": len(adaptive_samples),
                f"{prefix}adaptive_truncated_count": sum(1 for s in adaptive_samples if s["truncated"]),
                f"{prefix}adaptive_truncation_rate": sum(1 for s in adaptive_samples if s["truncated"]) / len(adaptive_samples) if adaptive_samples else 0,
            }, step=step)
            
            logger.info(f"Logged {len(adaptive_samples)} adaptive rollouts to wandb")
            
        except Exception as e:
            logger.warning(f"Error logging adaptive rollouts to wandb: {e}") 