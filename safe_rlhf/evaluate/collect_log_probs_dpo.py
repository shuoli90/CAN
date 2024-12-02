from dataclasses import dataclass, field
import torch
from typing import Optional, Dict
from accelerate import Accelerator
from peft import LoraConfig
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed, BitsAndBytesConfig
from trl import DPOTrainer
import argparse
import wandb
import random
import numpy as np
from torch.utils.data import random_split

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

def get_PKU(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24,
    prefer_safe=True,
) -> Dataset:
    """Load the stack-exchange-paired dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts are structured as follows:
      "Question: " + <prompt> + "\n\nAnswer: "
    """
    dataset = load_dataset(
        "PKU-Alignment/PKU-SafeRLHF-30K",
        cache_dir=cache_dir,
    )
    
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    
    train_dataset = dataset["test"]
    # split the train dataset into train and eval with 95% and 5% respectively
    # ds = train_dataset.train_test_split(test_size=0.1,)
    # train_dataset = ds['train']
    # eval_dataset = ds['test']

    original_columns = train_dataset.column_names

    def return_prompt_and_responses(samples) -> Dict[str, str]:
            response_0 = samples["response_0"]
            response_1 = samples["response_1"] 
            responses_list = list(zip(response_0, response_1))
            return {
                "prompt": [PROMPT_BEGIN + PROMPT_USER.format(input=question) + PROMPT_ASSISTANT for question in samples["prompt"]],
                # "chosen": [responses[safer_index] for responses, safer_index in zip(responses_list, indices)],
                "chosen": samples['response_0'],
                'rejected': samples['response_1'],
                # "rejected": [responses[1-safer_index] for responses, safer_index in zip(responses_list, indices)],
                "safer_response_id": samples["safer_response_id"],
                "better_response_id": samples["better_response_id"],
            }

    return train_dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.abspath(''), '..', '..'))
from safe_rlhf.models import AutoModelForScore, load_pretrained_models

safety_model, safety_tokenizer = load_pretrained_models(
        "PKU-Alignment/alpaca-7b-reproduced",
        model_max_length=512,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=True,
        peft_dir= os.path.join('/home/lishuo1/safe-rlhf/output', 'models', 'results_dpo_safety_3epoch'),
    )

testset = get_PKU()

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "out_proj",
        "fc_in",
        "fc_out",
        "wte",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

beta = 0.1
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,
    bf16=True,
    remove_unused_columns=False,
    run_name="dpo_llama_pku",
    output_dir="./output",
)

# 5. initialize the DPO trainer
dpo_trainer = DPOTrainer(
    safety_model,
    ref_model=None,
    args=training_args,
    beta=beta,
    train_dataset=testset,
    tokenizer=safety_tokenizer,
    peft_config=peft_config,
    max_prompt_length=512,
    max_length=512,
    precompute_ref_log_probs=True,
)

trainloader = dpo_trainer.get_train_dataloader()
safety_indices = trainloader.dataset.select_columns(['safer_response_id'])
better_indices = trainloader.dataset.select_columns(['better_response_id'])
safety_indices = safety_indices['safer_response_id']
helpful_indices = better_indices['better_response_id']
torch.save(safety_indices, 'safety_indices.pt')
torch.save(helpful_indices, 'helpful_indices.pt')

safety_0_log_probs = trainloader.dataset.select_columns(['reference_chosen_logps'])
safety_1_log_probs = trainloader.dataset.select_columns(['reference_rejected_logps'])
safety_0_log_probs = torch.tensor(safety_0_log_probs['reference_chosen_logps'])
safety_1_log_probs = torch.tensor(safety_1_log_probs['reference_rejected_logps'])
# save the log probabilities
torch.save(safety_0_log_probs, 'safety_0.1_0_log_probs.pt')
torch.save(safety_1_log_probs, 'safety_0.1_1_log_probs.pt')

del safety_model

help_model, help_tokenizer = load_pretrained_models(
        "PKU-Alignment/alpaca-7b-reproduced",
        model_max_length=512,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=True,
        peft_dir= os.path.join('/home/lishuo1/safe-rlhf/output', 'models', 'results_dpo_help_3epoch'),
    )
# 5. initialize the DPO trainer
dpo_trainer = DPOTrainer(
    help_model,
    ref_model=None,
    args=training_args,
    beta=beta,
    train_dataset=testset,
    tokenizer=safety_tokenizer,
    peft_config=peft_config,
    max_prompt_length=512,
    max_length=512,
    precompute_ref_log_probs=True,
)
trainloader = dpo_trainer.get_train_dataloader()
help_0_log_probs = torch.tensor(trainloader.dataset.select_columns(['reference_chosen_logps'])['reference_chosen_logps'])
help_1_log_probs = torch.tensor(trainloader.dataset.select_columns(['reference_rejected_logps'])['reference_rejected_logps'])
# save the log probabilities
torch.save(help_0_log_probs, 'help_0.1_0_log_probs.pt')
torch.save(help_1_log_probs, 'help_0.1_1_log_probs.pt')
del help_model

ref_model, ref_tokenizer = load_pretrained_models(
        "PKU-Alignment/alpaca-7b-reproduced",
        model_max_length=512,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=True,
    )
# 5. initialize the DPO trainer
dpo_trainer = DPOTrainer(
    ref_model,
    ref_model=None,
    args=training_args,
    beta=beta,
    train_dataset=testset,
    tokenizer=safety_tokenizer,
    peft_config=peft_config,
    max_prompt_length=512,
    max_length=512,
    precompute_ref_log_probs=True,
)
trainloader = dpo_trainer.get_train_dataloader()
ref_0_log_probs = torch.tensor(trainloader.dataset.select_columns(['reference_chosen_logps'])['reference_chosen_logps'])
ref_1_log_probs = torch.tensor(trainloader.dataset.select_columns(['reference_rejected_logps'])['reference_rejected_logps'])
# save the log probabilities
torch.save(ref_0_log_probs, 'ref_0_log_probs.pt')
torch.save(ref_1_log_probs, 'ref_1_log_probs.pt')
del ref_model

safety_001_model, safety_001_tokenizer = load_pretrained_models(
        "PKU-Alignment/alpaca-7b-reproduced",
        model_max_length=512,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=True,
        peft_dir= os.path.join('/home/lishuo1/safe-rlhf/output', 'models', 'dpo_safety_0.01'),
    )
# 5. initialize the DPO trainer
dpo_trainer = DPOTrainer(
    safety_001_model,
    ref_model=None,
    args=training_args,
    beta=0.01,
    train_dataset=testset,
    tokenizer=safety_tokenizer,
    peft_config=peft_config,
    max_prompt_length=512,
    max_length=512,
    precompute_ref_log_probs=True,
)
trainloader = dpo_trainer.get_train_dataloader()
safety_001_0_log_probs = torch.tensor(trainloader.dataset.select_columns(['reference_chosen_logps']))
safety_001_1_log_probs = torch.tensor(trainloader.dataset.select_columns(['reference_rejected_logps']))
# save the log probabilities
torch.save(safety_001_0_log_probs, 'safety_0.01_0_log_probs.pt')
torch.save(safety_001_1_log_probs, 'safety_0.01_1_log_probs.pt')


