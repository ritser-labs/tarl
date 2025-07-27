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
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.adaptive_trainer import TokenAdaptiveGRPOTrainer
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config


logger = logging.getLogger(__name__)


@dataclass
class AdaptiveGRPOScriptArguments(ScriptArguments):
    """
    Script arguments for Token-Adaptive GRPO training.
    """
    
    dataset_name: str = field(
        default="knoveleng/open-rs", metadata={"help": "Dataset name from the HuggingFace Hub"}
    )
    dataset_config: str = field(default=None, metadata={"help": "Dataset configuration name"})
    dataset_train_split: str = field(default="train", metadata={"help": "Dataset split for training"})
    dataset_test_split: str = field(default="test", metadata={"help": "Dataset split for evaluation"})
    
    # Reward function parameters
    reward_funcs: list[str] = field(
        default_factory=lambda: ["format", "cosine"], 
        metadata={"help": "Reward functions to use"}
    )
    
    # Cosine reward parameters
    cosine_min_value_wrong: float = field(default=-1.0, metadata={"help": "Minimum cosine reward for wrong answers"})
    cosine_max_value_wrong: float = field(default=-0.5, metadata={"help": "Maximum cosine reward for wrong answers"})
    cosine_min_value_correct: float = field(default=0.5, metadata={"help": "Minimum cosine reward for correct answers"})
    cosine_max_value_correct: float = field(default=1.0, metadata={"help": "Maximum cosine reward for correct answers"})
    cosine_max_len: int = field(default=3584, metadata={"help": "Maximum length for cosine reward scaling"})
    
    # Repetition penalty parameters
    repetition_n_grams: int = field(default=4, metadata={"help": "N-gram size for repetition penalty"})
    repetition_max_penalty: float = field(default=-1.0, metadata={"help": "Maximum repetition penalty"})
    
    # Code reward parameters
    code_language: str = field(default="python", metadata={"help": "Programming language for code format reward"})


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Log adaptive strategy parameters
    if training_args.use_token_adaptive:
        logger.info("*** Token-Adaptive Strategy Enabled ***")
        logger.info(f"  - Uncertainty threshold: {training_args.uncertainty_threshold}")
        logger.info(f"  - Entropy threshold: {training_args.entropy_threshold}")
        logger.info(f"  - Min prefix length: {training_args.min_prefix_length}")
        logger.info(f"  - Max resamples per prefix: {training_args.max_resamples_per_prefix}")
        logger.info(f"  - Adaptive temperature scale: {training_args.adaptive_temperature_scale}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    # Get reward functions
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": code_reward,
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # Format into conversation
    def make_conversation(example):
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        prompt.append({"role": "user", "content": example["problem"]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    logger.info("*** Initializing model kwargs ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    #########################################
    # Initialize the Token-Adaptive GRPO trainer
    #########################################
    logger.info("*** Initializing TokenAdaptiveGRPOTrainer ***")
    trainer = TokenAdaptiveGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Log final adaptive statistics
    if training_args.use_token_adaptive and trainer.adaptive_stats['total_rollouts'] > 0:
        logger.info("*** Final Adaptive Strategy Statistics ***")
        total_rollouts = trainer.adaptive_stats['total_rollouts']
        truncation_rate = trainer.adaptive_stats['truncated_rollouts'] / total_rollouts
        resample_rate = trainer.adaptive_stats['resampled_rollouts'] / total_rollouts
        avg_failure_point = (trainer.adaptive_stats['avg_failure_point'] / 
                            max(1, trainer.adaptive_stats['truncated_rollouts']))
        
        logger.info(f"  - Total rollouts: {total_rollouts}")
        logger.info(f"  - Truncation rate: {truncation_rate:.2%}")
        logger.info(f"  - Resample rate: {resample_rate:.2%}")
        logger.info(f"  - Average failure point: {avg_failure_point:.1f}")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1", "token-adaptive"],
    }
    
    # Add adaptive strategy info to model card
    if training_args.use_token_adaptive:
        kwargs["adaptive_strategy"] = {
            "uncertainty_threshold": training_args.uncertainty_threshold,
            "entropy_threshold": training_args.entropy_threshold,
            "min_prefix_length": training_args.min_prefix_length,
            "max_resamples_per_prefix": training_args.max_resamples_per_prefix,
            "adaptive_temperature_scale": training_args.adaptive_temperature_scale,
        }

    trainer.create_model_card(training_args.hub_model_id, **kwargs)


if __name__ == "__main__":
    parser = TrlParser((AdaptiveGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args) 