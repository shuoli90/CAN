import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dataclasses import dataclass, field
import torch
from typing import Optional, Dict
from accelerate import Accelerator
from peft import LoraConfig
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed, BitsAndBytesConfig
from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from trl import DPOTrainer
import argparse
import wandb
import random
import numpy as np
from torch.utils.data import random_split
import argparse
from torch.utils.data import DataLoader
import json
from tqdm import tqdm 
from torch.utils.data.distributed import DistributedSampler

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: '
PROMPT_ASSISTANT: str = ' ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT

def get_PKU(
    data_dir: str = "data/rl",
    sanity_check: bool = False,
    cache_dir: Optional[str] = None,
    num_proc=24,
    prefer_safe=True,
    tokenizer=None,
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
        "PKU-Alignment/PKU-SafeRLHF",
        cache_dir=cache_dir,
    )
    
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    original_columns = train_dataset.column_names

    def return_prompt_and_responses(samples) -> Dict[str, str]:
            prompts = [PROMPT_BEGIN + PROMPT_USER + prompt + PROMPT_ASSISTANT for prompt in samples["prompt"]]
            responses_0_all = [prompt+response for prompt, response in zip(prompts, samples["response_0"])]
            responses_1_all = [prompt+response for prompt, response in zip(prompts, samples["response_1"])] 
            responses_0 = samples["response_0"]
            responses_1 = samples["response_1"]
            return {
                "responses_0_all": responses_0_all,
                "responses_1_all": responses_1_all,
                "prompts": prompts,
                "responses_0": responses_0,
                "responses_1": responses_1,
            }

    return train_dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    ), eval_dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--cost_model_name_or_path", type=str, default="PKU-Alignment/beaver-7b-v1.0-cost")
    parser.add_argument("--reward_model_name_or_path", type=str, default="PKU-Alignment/beaver-7b-v1.0-reward")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--cost_device", type=int, default=6)
    parser.add_argument("--reward_device", type=int, default=7)
    args = parser.parse_args()

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        entity="lishuo1-penn",
        project="rlhf-dpo",
        # comment this line to turn on wandb
        mode="disabled",
    )

    cost_model, cost_tokenizer = load_pretrained_models(
        args.cost_model_name_or_path,
        model_max_length=args.max_length,
        padding_side='left',
        auto_model_type=AutoModelForScore,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        auto_device_mapping=True
    )
    cost_model.config.use_cache = False
    cost_model.eval()
   
    reward_model, reward_tokenizer = load_pretrained_models(
        args.reward_model_name_or_path,
        model_max_length=args.max_length,
        padding_side='left',
        auto_model_type=AutoModelForScore,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        auto_device_mapping=True
    )
    reward_model.config.use_cache = False
    reward_model.eval()

    trainset, evalset = get_PKU()
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, sampler=DistributedSampler(trainset, shuffle=True),)
    eval_loader = DataLoader(evalset, batch_size=args.batch_size, shuffle=False, sampler=DistributedSampler(evalset, shuffle=True),)

    prompts = []
    responses_0 = []
    responses_1 = []
    rewards_0 = []
    rewards_1 = []
    costs_0 = []
    costs_1 = []

    with torch.no_grad():
        loaders = {'train': train_loader, 'eval': eval_loader}
        for stage in ['train', 'eval']:
            loader = loaders[stage]
            for idx, batch in tqdm(enumerate(loader), total=len(loader)):
                full_0_s_ids = reward_tokenizer.batch_encode_plus(
                    batch['responses_0_all'],
                    return_tensors='pt',
                    padding='max_length',
                    max_length=args.max_length,
                )['input_ids']
                full_1_s_ids = reward_tokenizer.batch_encode_plus(
                    batch['responses_1_all'],
                    return_tensors='pt',
                    padding='max_length',
                    max_length=args.max_length,
                    truncation=True,
                )['input_ids']

                attention_mask_0 = torch.logical_and(
                    full_0_s_ids.not_equal(reward_tokenizer.pad_token_id),
                    full_0_s_ids.not_equal(reward_tokenizer.unk_token_id),
                )

                attention_mask_1 = torch.logical_and(
                    full_1_s_ids.not_equal(reward_tokenizer.pad_token_id),
                    full_1_s_ids.not_equal(reward_tokenizer.unk_token_id),
                )

                reward_0 = reward_model(full_0_s_ids.to(reward_model.device), attention_mask=attention_mask_0.to(reward_model.device)).end_scores
                reward_1 = reward_model(full_1_s_ids.to(reward_model.device), attention_mask=attention_mask_1.to(reward_model.device)).end_scores
                cost_0 = cost_model(full_0_s_ids.to(cost_model.device), attention_mask=attention_mask_0.to(cost_model.device)).end_scores
                cost_1 = cost_model(full_1_s_ids.to(cost_model.device), attention_mask=attention_mask_1.to(cost_model.device)).end_scores

                prompts.extend(batch['prompts'])
                responses_0.extend(batch['responses_0'])
                responses_1.extend(batch['responses_1'])
                rewards_0.extend(reward_0.cpu().tolist())
                rewards_1.extend(reward_1.cpu().tolist())
                costs_0.extend(cost_0.cpu().tolist())
                costs_1.extend(cost_1.cpu().tolist())

                if idx % 10 == 0:
                    with open(f'{stage}_v1.json', 'a') as f:
                        json.dump({
                            'prompts': prompts,
                            'responses_0': responses_0,
                            'responses_1': responses_1,
                            'rewards_0': rewards_0,
                            'rewards_1': rewards_1,
                            'costs_0': costs_0,
                            'costs_1': costs_1,
                        }, f)

