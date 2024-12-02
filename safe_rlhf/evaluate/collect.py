# Copyright 2023-2024 PKU-Alignment Team. All Rights Reserved.
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
# ==============================================================================

from __future__ import annotations

import argparse
import csv
import os
from typing import NamedTuple

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.utils import is_torch_bf16_gpu_available, is_torch_tf32_available
import datasets
import json

from safe_rlhf.configs import get_deepspeed_eval_config
from safe_rlhf.datasets import PromptOnlyDataset, SupervisedDataset, parse_dataset
from safe_rlhf.logger import set_logger_level
from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from safe_rlhf.utils import (
    batch_retokenize,
    is_main_process,
    is_same_tokenizer,
    seed_everything,
    str2bool,
    to_device,
)

from torch.utils.data import TensorDataset

def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='deepspeed --module safe_rlhf.evaluate.score',
        description='Score the performance of a model.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model
    model_parser = parser.add_argument_group('model')
    model_parser.add_argument(
        '--cost_model_name_or_path',
        type=str,
        help='the name or path of the cost model to load from',
        required=True,
    )
    model_parser.add_argument(
        '--reward_model_name_or_path',
        type=str,
        help='the name or path of the reward model to load from',
        required=True,
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='The maximum sequence length of the model.',
    )
    model_parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=8,
        help='The number of sequences to be returned.',
    )
    model_parser.add_argument(
        '--trust_remote_code',
        type=str2bool,
        default=False,
        help='Whether to trust the remote code.',
    )

    # Evaluation
    evaluation_parser = parser.add_argument_group('evaluation')
    evaluation_parser.add_argument(
        '--per_device_eval_batch_size',
        type=int,
        default=8,
        help='Batch size (per device) for the evaluation dataloader.',
    )
    evaluation_parser.add_argument(
        '--total_generate_size',
        type=int,
        default=3000,
        help='The total amount of prompts to be completed.',
    )
    evaluation_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible evaluation.',
    )
    evaluation_parser.add_argument(
        '--fp16',
        type=str2bool,
        default=False,
        help='Whether to use float16 precision.',
    )
    evaluation_parser.add_argument(
        '--bf16',
        type=str2bool,
        default=False,
        help='Whether to use bfloat16 precision.',
    )
    evaluation_parser.add_argument(
        '--tf32',
        type=str2bool,
        default=None,
        help='Whether to use tf32 mix precision.',
    )

    # Logging
    logging_parser = parser.add_argument_group('logging')
    logging_parser.add_argument(
        '--output_dir',
        type=str,
        default='output/score',
        help='Where to store the evaluation output.',
    )
    logging_parser.add_argument(
        '--input_dir',
        type=str,
        default='output/generate',
        help='Where to store the evaluation output.',
    )

    # DeepSpeed
    deepspeed_parser = parser.add_argument_group('deepspeed')
    deepspeed_parser.add_argument(
        '--local_rank',
        type=int,
        default=-1,
        help='Local rank for distributed training on GPUs',
    )
    deepspeed_parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help='ZeRO optimization stage for models.',
    )
    deepspeed_parser.add_argument(
        '--offload',
        type=str,
        default='none',
        choices=['none', 'parameter', 'optimizer', 'all'],
        help='Offload parameters and/or optimizer states to CPU.',
    )
    parser = deepspeed.add_config_arguments(parser)

    args = parser.parse_args()
    if args.local_rank == -1:
        parser.error('`local_rank` not set, please use DeepSpeed launcher to run this script.')
    if args.fp16 and args.bf16:
        parser.error('Cannot use both bf16 and fp16 precision.')
    if args.bf16 and not is_torch_bf16_gpu_available():
        parser.error(
            'bf16 precision is not supported on this GPU. '
            'Please disable `--bf16` flag or use another precision flag (e.g., `--fp16`).',
        )
    if args.tf32 is not None and is_torch_tf32_available():
        torch.backends.cuda.matmul.allow_tf32 = args.tf32

    return args

def prompt_template(prompt: str, response: str) -> str:
    return f'Prompt: {prompt}\n\nResponse: {response}'

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


def batch_scoring(
    batch: list[torch.Tensor],
    reward_model: PreTrainedModel,
    cost_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
):
    prompt = batch['prompt']
    prompt_ids = tokenizer.batch_encode_plus(
        prompt,
        return_tensors='pt',
        padding='max_length',
        max_length=args.max_length,
        truncation=True,
    )['input_ids']
    response_0_ids = tokenizer.batch_encode_plus(
        batch['response_0'],
        return_tensors='pt',
        padding='max_length',
        max_length=args.max_length,
        truncation=True,
    )['input_ids']
    response_1_ids = tokenizer.batch_encode_plus(
        batch['response_1'],
        return_tensors='pt',
        padding='max_length',
        max_length=args.max_length,
        truncation=True,
    )['input_ids']

    safe_indices = to_device(torch.tensor(batch['safer_response_id']).reshape(-1, 1), args.device)
    help_indices = to_device(torch.tensor(batch['better_response_id']).reshape(-1, 1), args.device)

    prompt_ids = to_device(prompt_ids, args.device)
    response_0_ids = to_device(response_0_ids, args.device)
    response_1_ids = to_device(response_1_ids, args.device)
    full_0_s = [PROMPT_BEGIN + PROMPT_USER.format(input=question) + PROMPT_ASSISTANT + ' ' + response for question, response in zip(batch['prompt'], batch['response_0'])]
    full_1_s = [PROMPT_BEGIN + PROMPT_USER.format(input=question) + PROMPT_ASSISTANT + ' ' + response for question, response in zip(batch['prompt'], batch['response_1'])]
    full_0_s_ids = tokenizer.batch_encode_plus(
        full_0_s,
        return_tensors='pt',
        padding='max_length',
        max_length=args.max_length,
        truncation=True,
    )['input_ids']
    full_1_s_ids = tokenizer.batch_encode_plus(
        full_1_s,
        return_tensors='pt',
        padding='max_length',
        max_length=args.max_length,
        truncation=True,
    )['input_ids']
    full_0_s_ids = to_device(full_0_s_ids, args.device)
    full_1_s_ids = to_device(full_1_s_ids, args.device)
    dist.barrier()
    
    attention_mask_0 = torch.logical_and(
        full_0_s_ids.not_equal(tokenizer.pad_token_id),
        full_0_s_ids.not_equal(tokenizer.unk_token_id),
    )

    attention_mask_1 = torch.logical_and(
        full_1_s_ids.not_equal(tokenizer.pad_token_id),
        full_1_s_ids.not_equal(tokenizer.unk_token_id),
    )
    
    with torch.no_grad():
        reward_score_0 = reward_model(
            input_ids=full_0_s_ids,
            attention_mask=attention_mask_0,
        ).end_scores

        reward_score_1 = reward_model(
            input_ids=full_1_s_ids,
            attention_mask=attention_mask_1,
        ).end_scores

        cost_score_0 = cost_model(
            input_ids=full_0_s_ids,
            attention_mask=attention_mask_0,
        ).end_scores

        cost_score_1 = cost_model(
            input_ids=full_1_s_ids,
            attention_mask=attention_mask_1,
        ).end_scores
        
    if is_main_process():
        gathered_reward_scores_0 = [
            torch.empty_like(reward_score_0) for _ in range(dist.get_world_size())
        ]
        gathered_reward_scores_1 = [
            torch.empty_like(cost_score_1) for _ in range(dist.get_world_size())
        ]
        gathered_cost_scores_0 = [
            torch.empty_like(reward_score_0) for _ in range(dist.get_world_size())
        ]
        gathered_cost_scores_1 = [
            torch.empty_like(cost_score_1) for _ in range(dist.get_world_size())
        ]
        gathered_prompt_ids = [
            torch.empty_like(prompt_ids) for _ in range(dist.get_world_size())
        ]
        gathered_response_0_ids = [
            torch.empty_like(response_0_ids) for _ in range(dist.get_world_size())
        ]
        gathered_response_1_ids = [
            torch.empty_like(response_1_ids) for _ in range(dist.get_world_size())
        ]
        gathered_safe_indices = [
            torch.empty_like(safe_indices) for _ in range(dist.get_world_size())
        ]
        gathered_help_indices = [
            torch.empty_like(help_indices) for _ in range(dist.get_world_size())
        ]
    else:
        gathered_reward_scores_0 = []
        gathered_reward_scores_1 = []
        gathered_cost_scores_0 = []
        gathered_cost_scores_1 = []
        gathered_prompt_ids = []
        gathered_response_0_ids = []
        gathered_response_1_ids = []
        gathered_safe_indices = []
        gathered_help_indices = []


    dist.gather(reward_score_0, gathered_reward_scores_0, dst=0)
    dist.gather(reward_score_1, gathered_reward_scores_1, dst=0)
    dist.gather(cost_score_0, gathered_cost_scores_0, dst=0)
    dist.gather(cost_score_1, gathered_cost_scores_1, dst=0)
    dist.gather(prompt_ids, gathered_prompt_ids, dst=0)
    dist.gather(response_0_ids, gathered_response_0_ids, dst=0)
    dist.gather(response_1_ids, gathered_response_1_ids, dst=0)
    dist.gather(safe_indices, gathered_safe_indices, dst=0)
    dist.gather(help_indices, gathered_help_indices, dst=0)

    if is_main_process():
        gathered_reward_scores_0 = torch.cat(gathered_reward_scores_0, dim=0)
        gathered_reward_scores_1 = torch.cat(gathered_reward_scores_1, dim=0)
        gathered_cost_scores_0 = torch.cat(gathered_cost_scores_0, dim=0)
        gathered_cost_scores_1 = torch.cat(gathered_cost_scores_1, dim=0)
        gathered_prompt_ids = torch.cat(gathered_prompt_ids, dim=0)
        gathered_response_0_ids = torch.cat(gathered_response_0_ids, dim=0)
        gathered_response_1_ids = torch.cat(gathered_response_1_ids, dim=0)
        gathered_safe_indices = torch.cat(gathered_safe_indices, dim=0)
        gathered_help_indices = torch.cat(gathered_help_indices, dim=0)

    return gathered_reward_scores_0, gathered_reward_scores_1, gathered_cost_scores_0, gathered_cost_scores_1, gathered_prompt_ids, gathered_response_0_ids, gathered_response_1_ids, gathered_safe_indices, gathered_help_indices

def main() -> None:  # pylint: disable=too-many-locals,too-many-statements
    args = parse_arguments()

    deepspeed.init_distributed()

    args.global_rank = dist.get_rank()
    args.device = torch.device('cuda', args.local_rank)
    torch.cuda.set_device(args.device)
    seed_everything(args.seed)
    set_logger_level()

    dist.barrier()

    ds_config = get_deepspeed_eval_config(
        stage=args.zero_stage,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    if ds_config['zero_optimization']['stage'] == 3:
        args.dschf = HfDeepSpeedConfig(ds_config)

    cost_model, cost_tokenizer = load_pretrained_models(
        args.cost_model_name_or_path,
        model_max_length=args.max_length,
        padding_side='left',
        auto_model_type=AutoModelForScore,
        trust_remote_code=args.trust_remote_code
    )
   
    reward_model, reward_tokenizer = load_pretrained_models(
        args.reward_model_name_or_path,
        model_max_length=args.max_length,
        padding_side='left',
        auto_model_type=AutoModelForScore,
        trust_remote_code=args.trust_remote_code
    )

    dataset = datasets.load_dataset(
        "PKU-Alignment/PKU-SafeRLHF-30K",
        split='test',
    )

    dataloader = DataLoader(
        dataset,
        sampler=DistributedSampler(dataset, shuffle=True),
        batch_size=args.per_device_eval_batch_size,
    )

    if is_main_process:
        res_score = None
    res_size = 0

    cost_model, *_ = deepspeed.initialize(model=cost_model, config=ds_config)
    cost_model.eval()

    reward_model, *_ = deepspeed.initialize(model=reward_model, config=ds_config)
    reward_model.eval()

    dist.barrier()

    # Scoring
    for i, batch in enumerate(
        tqdm(
            dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
        ),
        start=0,
    ):  
        reward_scores_0, reward_scores_1, cost_scores_0, cost_scores_1, prompt_ids, response_0_ids, response_1_ids, safe_indices, help_indices = batch_scoring(
            batch,
            reward_model=reward_model,
            cost_model=cost_model,
            tokenizer=reward_tokenizer,
            args=args,
        ) 

        if is_main_process():
            
            if i == 0:
                reward_scores_res_0 = reward_scores_0
                reward_scores_res_1 = reward_scores_1
                cost_scores_res_0 = cost_scores_0
                cost_scores_res_1 = cost_scores_1
                prompt_ids_res = prompt_ids
                response_0_ids_res = response_0_ids
                response_1_ids_res = response_1_ids
                safe_indices_res = safe_indices
                help_indices_res = help_indices
            else:
                reward_scores_res_0 = torch.cat((reward_scores_res_0, reward_scores_0), dim=0)
                reward_scores_res_1 = torch.cat((reward_scores_res_1, reward_scores_1), dim=0)
                cost_scores_res_0 = torch.cat((cost_scores_res_0, cost_scores_0), dim=0)
                cost_scores_res_1 = torch.cat((cost_scores_res_1, cost_scores_1), dim=0)
                prompt_ids_res = torch.cat((prompt_ids_res, prompt_ids), dim=0)
                response_0_ids_res = torch.cat((response_0_ids_res, response_0_ids), dim=0)
                response_1_ids_res = torch.cat((response_1_ids_res, response_1_ids), dim=0)
                safe_indices_res = torch.cat((safe_indices_res, safe_indices), dim=0)
                help_indices_res = torch.cat((help_indices_res, help_indices), dim=0)

            if i % 20 == 0:
                results = {
                    'reward_scores_0': reward_scores_res_0.cpu().numpy().tolist(),
                    'reward_scores_1': reward_scores_res_1.cpu().numpy().tolist(),
                    'cost_scores_0': cost_scores_res_0.cpu().numpy().tolist(),
                    'cost_scores_1': cost_scores_res_1.cpu().numpy().tolist(),
                    'prompt_ids': prompt_ids_res.cpu().numpy().tolist(),
                    'response_0_ids': response_0_ids_res.cpu().numpy().tolist(),
                    'response_1_ids': response_1_ids_res.cpu().numpy().tolist(),
                    'safe_indices': safe_indices_res.cpu().numpy().tolist(),
                    'help_indices': help_indices_res.cpu().numpy().tolist()
                }
                # save results to a json file
                with open(args.output_dir+f'/PKU-SafeRLHF-30K_test_v3.json', 'w') as f:
                    json.dump(results, f)

if __name__ == '__main__':
    main()
