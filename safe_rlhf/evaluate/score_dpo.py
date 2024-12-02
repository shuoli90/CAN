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

from safe_rlhf.configs import get_deepspeed_eval_config
from safe_rlhf.datasets import PromptOnlyDataset, parse_dataset
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
        '--model_name_or_path',
        type=str,
        help='the name or path of the model to load from',
        required=True,
    )
    model_parser.add_argument(
        '--reward_model_name_or_path',
        type=str,
        help='the name or path of the reward model to load from',
        required=True,
    )
    model_parser.add_argument(
        '--file_name',
        type=str,
        help='the name of the file to load from',
        required=True,
    )
    model_parser.add_argument(
        '--cost_model_name_or_path',
        type=str,
        help='the name or path of the cost model to load from',
        required=True,
    )
    model_parser.add_argument(
        '--max_length',
        type=int,
        default=512,
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
    model_parser.add_argument(
        '--peft_dir',
        type=str,
        default=None,
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


def batch_scoring(
    batch: list[torch.Tensor],
    reward_model: PreTrainedModel,
    cost_model: PreTrainedModel,
    reward_tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
):
    pair_ids = to_device(batch[0], args.device).to(torch.int64)
    dist.barrier()
    
    attention_mask = torch.logical_and(
        pair_ids.not_equal(reward_tokenizer.pad_token_id),
        pair_ids.not_equal(reward_tokenizer.unk_token_id),
    )
    
    with torch.no_grad():
        reward_score = reward_model(
            pair_ids,
            attention_mask=attention_mask,
        ).end_scores
        # reward_score = reward_model(pair_ids, attention_mask=attention_mask).end_scores

        cost_score = cost_model(
            pair_ids,
            attention_mask=attention_mask,
        ).end_scores

    reward_score = torch.hstack((pair_ids, reward_score, cost_score)) # create the demo-score pairs

    if is_main_process():
        gathered_reward_scores = [
            torch.empty_like(reward_score) for _ in range(dist.get_world_size())
        ]
    else:
        gathered_reward_scores = []

    dist.gather(reward_score, gathered_reward_scores, dst=0)
    if is_main_process():
        gathered_reward_scores = torch.cat(gathered_reward_scores, dim=0)

    return gathered_reward_scores


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
   
    reward_model, reward_tokenizer = load_pretrained_models(
        args.reward_model_name_or_path,
        model_max_length=args.max_length,
        padding_side='left',
        auto_model_type=AutoModelForScore,
        trust_remote_code=args.trust_remote_code
    )

    cost_model, cost_tokenizer = load_pretrained_models(
        args.cost_model_name_or_path,
        model_max_length=args.max_length,
        padding_side='left',
        auto_model_type=AutoModelForScore,
        trust_remote_code=args.trust_remote_code
    )

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path))
    # output_file_path = args.input_dir+"_dpo"+f'/{model_name}_{args.peft_dir}_output_ids.pt' # for test purpuse
    output_file_path = args.input_dir+"_dpo"+f'/{args.file_name}.pt'

    if is_main_process:
        res_score = None
    res_size = 0

    reward_model, *_ = deepspeed.initialize(model=reward_model, config=ds_config)
    reward_model.eval()

    cost_model, *_ = deepspeed.initialize(model=cost_model, config=ds_config)
    cost_model.eval()

    output_ids = torch.load(output_file_path, map_location='cpu')
    res = None
    output_ids = output_ids.view(-1, args.max_length)[:, :args.max_length]

    dataset = TensorDataset(output_ids)
    dataloader = DataLoader( # may only works for a single machine
        dataset,
        sampler=DistributedSampler(dataset, shuffle=False),
        batch_size=args.per_device_eval_batch_size,
    )

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
        scores = batch_scoring(
            batch,
            reward_model=reward_model,
            cost_model=cost_model,
            reward_tokenizer=reward_tokenizer,
            args=args,
        )    

        if is_main_process():
            scores = scores.view(1, -1, args.max_length + 2)
            if i == 0 and res is None:
                res = scores
            else:
                res = torch.cat((res, scores), dim=0)

            # torch.save(res, args.output_dir+f'/{model_name}_{args.peft_dir}_scores_test.pt')
            torch.save(res, args.output_dir+f'/{args.file_name}_scores_test.pt')

    if is_main_process():
        # torch.save(res, args.output_dir+f'/{model_name}_{args.peft_dir}_scores_teset.pt')
        torch.save(res, args.output_dir+f'/{args.file_name}_scores_test.pt')


if __name__ == '__main__':
    main()
