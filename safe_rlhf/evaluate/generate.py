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



def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        prog='deepspeed --module safe_rlhf.evaluate.generate',
        description='Obtain the generation of a model.',
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
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
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
        help='The directory of the PEFT adapter to load.',
    )

    # Dataset
    dataset_parser = parser.add_argument_group('dataset')
    dataset_parser.add_argument(
        '--datasets',
        type=parse_dataset,
        nargs='+',
        metavar='DATASET[:PROPORTION[:PATH]]',
        help='Dataset name(s) registered in the raw dataset.',
        required=True,
    )
    dataset_parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=8,
        help='The number of sequences to return.',
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
        default=None,
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

def batch_generation(
    batch: dict[str, torch.Tensor],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    num_return_sequences: int,
    args: argparse.Namespace,
):
    batch = to_device(batch, args.device)
    seed = args.seed + dist.get_rank()
    seed_everything(seed)
    with torch.no_grad():
        input_ids=batch['input_ids']
        output_ids = model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_length=args.max_length,
            synced_gpus=True,
            do_sample=True,
            num_return_sequences=num_return_sequences,
        )

    dist.barrier()

    # Gather all output_ids and scores
    pad_length = args.max_length - output_ids.size(-1) # force the unified max length of output_ids
    if pad_length > 0:
        output_ids = F.pad(
            output_ids,
            (0, pad_length),
            mode='constant',
            value=tokenizer.unk_token_id,
        )
    output_ids = output_ids.view(
        input_ids.size(0),
        num_return_sequences,
        -1,
    )[:, :, :args.max_length]

    if is_main_process():
        gathered_output_ids = [torch.empty_like(output_ids) for _ in range(dist.get_world_size())]
    else:
        gathered_output_ids = []

    dist.gather(output_ids, gathered_output_ids, dst=0)

    if is_main_process():
        gathered_output_ids = torch.cat(gathered_output_ids, dim=0)
        
    return  gathered_output_ids


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

    model, tokenizer = load_pretrained_models(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=args.trust_remote_code,
        # peft_dir= os.path.join(args.output_dir, 'models', args.peft_dir) if args.peft_dir else None,
        peft_dir = os.path.join('/home/lishuo1/safe-rlhf/safe_rlhf/algorithms/cdpo', args.peft_dir) if args.peft_dir else None,
    )

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path))

    if is_main_process:
        res_input = None
        res_output = None
        res_size = 0

    model, *_ = deepspeed.initialize(model=model, config=ds_config)
    model.eval()

    dataset = PromptOnlyDataset(args.datasets, tokenizer)
    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.get_collator(),
        sampler=DistributedSampler(dataset, shuffle=False),
        batch_size=args.per_device_eval_batch_size,
    )
    num_batches = len(dataloader)

    dist.barrier()

    # Generation
    for i, batch in enumerate(
        tqdm(
            dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
        ),
        start=0
    ):
        generated_output_ids = batch_generation(
            batch,
            model=model,
            tokenizer=tokenizer,
            args=args,
            num_return_sequences=args.num_return_sequences,
        )

        if is_main_process():
            if i == 0:
                res_output = generated_output_ids
            else:
                res_output = torch.cat((res_output, generated_output_ids), dim=0)

            if i % 20 == 0:
                if args.peft_dir is not None:
                    torch.save(res_output, args.output_dir+f'/generate_dpo/{model_name}_{args.peft_dir}_output_ids.pt')
                else:
                    torch.save(res_output, args.output_dir+f'/generate_dpo/{model_name}_output_ids.pt')
    
    if is_main_process():
        if args.peft_dir is not None:
            torch.save(res_output, args.output_dir+f'/generate_dpo/{model_name}_{args.peft_dir}_output_ids.pt')
        else:
            torch.save(res_output, args.output_dir+f'/generate_dpo/{model_name}_output_ids.pt')

if __name__ == '__main__':
    main()
