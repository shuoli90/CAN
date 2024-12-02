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
from datasets import Dataset, load_dataset
from typing import Dict, Optional

from safe_rlhf.configs import get_deepspeed_eval_config
from safe_rlhf.datasets import PromptOnlyDataset, SupervisedDataset, parse_dataset
from safe_rlhf.logger import set_logger_level
from safe_rlhf.models import AutoModelForScore, load_pretrained_models
from safe_rlhf.datasets.utils import left_padding, right_padding
from safe_rlhf.utils import (
    batch_retokenize,
    is_main_process,
    is_same_tokenizer,
    seed_everything,
    str2bool,
    to_device,
    gather_log_probabilities,
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
        help='the name or path of the SFT model to load from',
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
    model_parser.add_argument(
        "--safety_peft_dir",
        type=str,
        # default='results_dpo_safety_3epoch',
        default='dpo_safety_0.025',
        help='The directory of the safety model PEFT results.',
    )
    model_parser.add_argument(
        "--help_peft_dir",
        type=str,
        # default='results_dpo_help_3epoch',
        default='dpo_help_0.025',
        help='The directory of the help model PEFT results.',
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


def batch_scoring(
    batch: list[torch.Tensor],
    safety_model: PreTrainedModel,
    help_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    num_return_sequences: int,
):  
    with torch.no_grad():
        prompts_ids = to_device(batch['prompt_ids'], args.device)
        responses_0_ids = to_device(batch['responses_0_ids'], args.device)
        responses_1_ids = to_device(batch['responses_1_ids'], args.device)

        # remove eos token from prompts
        prompts_ids = prompts_ids[:, :-1]
        # remove bos token from responses
        responses_0_ids = responses_0_ids[:, 1:]
        responses_1_ids = responses_1_ids[:, 1:]
        # concatenate prompts and responses
        responses_0_all = torch.cat([prompts_ids, responses_0_ids], dim=-1)
        responses_1_all = torch.cat([prompts_ids, responses_1_ids], dim=-1)

        responses_0_mask = torch.logical_and(
            responses_0_all.not_equal(tokenizer.pad_token_id),
            responses_0_all.not_equal(tokenizer.unk_token_id),
        )
        responses_1_mask = torch.logical_and(
            responses_1_all.not_equal(tokenizer.pad_token_id),
            responses_1_all.not_equal(tokenizer.unk_token_id),
        )

        start = prompts_ids.size(1) - 1

        safety_0_logits = safety_model(responses_0_all).logits
        safety_0_log_probs = gather_log_probabilities(safety_0_logits[:, start:-1], responses_0_all[:, start+1:], responses_0_mask[:, start+1:])

        safety_1_logits = safety_model(responses_1_all).logits
        safety_1_log_probs = gather_log_probabilities(safety_1_logits[:, start:-1], responses_1_all[:, start+1:], responses_1_mask[:, start+1:])

        help_0_logits = help_model(responses_0_all).logits
        help_0_log_probs = gather_log_probabilities(help_0_logits[:, start:-1], responses_0_all[:, start+1:], responses_0_mask[:, start+1:])

        help_1_logits = help_model(responses_1_all).logits
        help_1_log_probs = gather_log_probabilities(help_1_logits[:, start:-1], responses_1_all[:, start+1:], responses_1_mask[:, start+1:])
    dist.barrier()

    # Gather all output_ids and scores
    pad_length = args.max_length - safety_0_log_probs.size(-1) # force the unified max length of output_ids
    if pad_length > 0:
        safety_0_log_probs = F.pad(
            safety_0_log_probs,
            (0, pad_length),
            mode='constant',
            value=10.0,
        )
    
    pad_length = args.max_length - safety_1_log_probs.size(-1)
    if pad_length > 0:
        safety_1_log_probs = F.pad(
            safety_1_log_probs,
            (0, pad_length),
            mode='constant',
            value=10.0,
        )

    pad_length = args.max_length - help_0_log_probs.size(-1)
    if pad_length > 0:
        help_0_log_probs = F.pad(
            help_0_log_probs,
            (0, pad_length),
            mode='constant',
            value=10.0,
        )

    pad_length = args.max_length - help_1_log_probs.size(-1)
    if pad_length > 0:
        help_1_log_probs = F.pad(
            help_1_log_probs,
            (0, pad_length),
            mode='constant',
            value=10.0,
        )

    outputs = torch.cat([safety_0_log_probs, safety_1_log_probs, help_0_log_probs, help_1_log_probs], dim=1).view(-1, 4 * args.max_length)
    dist.barrier()  
        
    if is_main_process():
        gathered = [
            torch.empty_like(outputs) for _ in range(dist.get_world_size())
        ]
    else:
        gathered = []
    dist.gather(outputs, gathered, dst=0)
    if is_main_process():
        gathered = torch.cat(gathered, dim=0)
    return gathered

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
        "PKU-Alignment/PKU-SafeRLHF-30K",
        cache_dir=cache_dir,
    )
    
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))
    
    train_dataset = dataset["train"].select(range(min(len(dataset), 1000)))

    original_columns = train_dataset.column_names

    def return_prompt_and_responses(samples) -> Dict[str, str]:
        prompts = [PROMPT_BEGIN + PROMPT_USER.format(input=question) + PROMPT_ASSISTANT for question in samples["prompt"]]
        prompts_ids = tokenizer.batch_encode_plus(prompts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)["input_ids"]
        responses_0_full = [prompt+response for prompt, response in zip(prompts, samples['response_0'])]
        responses_1_full = [prompt+response for prompt, response in zip(prompts, samples['response_1'])]
        responses_0_ids = tokenizer.batch_encode_plus(responses_0_full, padding=True, truncation=True, return_tensors="pt")["input_ids"]
        responses_1_ids = tokenizer.batch_encode_plus(responses_1_full, padding=True, truncation=True, return_tensors="pt")["input_ids"]

        return {
            "prompt_ids": prompts_ids,
            "responses_0_ids": responses_0_ids,
            "responses_1_ids": responses_1_ids,
        }

    return train_dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
    )

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

    safety_model, safety_tokenizer = load_pretrained_models(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=args.trust_remote_code,
        peft_dir= os.path.join(args.output_dir, 'models', args.safety_peft_dir),
    )

    help_model, help_tokenizer = load_pretrained_models(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side='left',
        auto_model_type=AutoModelForCausalLM,
        trust_remote_code=args.trust_remote_code,
        peft_dir= os.path.join(args.output_dir, 'models', args.help_peft_dir),
    )

    dist.barrier()
    trainset = get_PKU(tokenizer=safety_tokenizer)

    def new_collator(batch):
        return {
            'prompt_ids': left_padding([torch.tensor(sample['prompt_ids']) for sample in batch], padding_value=safety_tokenizer.pad_token_id),
            'responses_0_ids': right_padding([torch.tensor(sample['responses_0_ids']) for sample in batch], padding_value=safety_tokenizer.pad_token_id),
            'responses_1_ids': right_padding([torch.tensor(sample['responses_1_ids']) for sample in batch], padding_value=safety_tokenizer.pad_token_id),
        }

    dataloader = DataLoader( # may only works for a single machine
        trainset,
        collate_fn=new_collator,
        sampler=DistributedSampler(trainset, shuffle=False),
        batch_size=args.per_device_eval_batch_size,
    )

    ds_config = get_deepspeed_eval_config(
        stage=args.zero_stage,
        fp16=args.fp16,
        bf16=args.bf16,
    )

    if ds_config['zero_optimization']['stage'] == 3:
        args.dschf = HfDeepSpeedConfig(ds_config)

    safety_model, *_ = deepspeed.initialize(model=safety_model, config=ds_config)
    safety_model.eval()

    help_model, *_ = deepspeed.initialize(model=help_model, config=ds_config)
    help_model.eval()

    dist.barrier()
    res = None
    # Scoring
    for i, batch in enumerate(
        tqdm(
            dataloader,
            desc='Evaluating',
            disable=not is_main_process(),
        ),
        start=0,
    ):  
        gathered = batch_scoring(
            batch,
            safety_model=safety_model,
            help_model=help_model,
            tokenizer=safety_tokenizer,
            args=args,
            num_return_sequences=args.per_device_eval_batch_size,
        ) 

        if is_main_process():
            if i == 0 and res == None:
                res = gathered
            else:
                res = torch.cat((res, gathered), dim=0)

            if i % 20 == 0:
                torch.save(res, args.output_dir+f'/log_probs_dpo_modified_0025.pt')

    torch.save(res, args.output_dir+f'/log_probs_dpo_modified_0025.pt')

if __name__ == '__main__':
    main()
