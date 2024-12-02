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
import json
import os
import time

import openai
import torch
from tqdm import tqdm

from transformers import AutoModelForCausalLM

from safe_rlhf.configs.constants import PROMPT_INPUT
from safe_rlhf.models import load_pretrained_models
from safe_rlhf.utils import to_device, seed_everything


def parse_arguments() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate models with gpt4',
    )

    # Model
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help='the name or path of the first model (champion) in the arena to load from',
        required=True,
    )

    parser.add_argument(
        '--peft_dir',
        type=str,
        default=None,
        help='The directory of the PEFT adapter to load.',
    )

    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='The maximum sequence length of the model.',
    )

    parser.add_argument(
        '--device',
        type=int,
        default=3,
        help='The GPU device.',
    )

    parser.add_argument(
        '--num_return_sequences',
        type=int,
        default=1,
        help='The number of sequences to return.',
    )

    parser.add_argument(
        '--mode',
        type=str,
        help='the mode of comparison',
        required=True,
        default='safety',
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='A seed for reproducible evaluation.',
    )

    # Logging
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Where to store the eval output.',
    )

    return parser.parse_args()


def generate_answer(problems: list[str], model, tokenizer, args) -> list[str]:
    seed = args.seed
    seed_everything(seed)

    answers = []
    print(f'Generating answers with {args.model_name_or_path}')
    for problem in tqdm(problems):
        if args.mode == 'safety':
            prompt = PROMPT_INPUT.format(input=problem['prompt'])
        elif args.mode == 'help':
            prompt = PROMPT_INPUT.format(input=problem['instruction'])
        input_ids = to_device(
            tokenizer(prompt, return_tensors='pt'),
            device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else None)
        )
        output_ids = model.generate(
            **input_ids,
            max_length=args.max_length,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)[len(prompt):]
        answers.append(answer)
    return answers

def main() -> None:
    """The main function."""
    args = parse_arguments()

    PROBLEM_PATH = os.path.join(os.path.dirname(__file__), 'problem_'+args.mode+'.json')
    with open(PROBLEM_PATH, encoding='utf-8') as f:
        problems = json.load(f)

    model, tokenizer = load_pretrained_models(
        args.model_name_or_path,
        auto_device_mapping="cuda:"+str(args.device),
        trust_remote_code=True,
        model_max_length=args.max_length,
        auto_model_type=AutoModelForCausalLM,
        peft_dir= os.path.join('output/models', args.peft_dir) if args.peft_dir else None,
    )
    model_name = os.path.basename(os.path.normpath(args.model_name_or_path))

    model.eval()
    with torch.no_grad():
        answers = generate_answer(problems, model, tokenizer, args)

    if args.peft_dir is not None:
        with open(os.path.join(args.output_dir, f'{(args.peft_dir).replace("/", "_")}_gpt-'+args.mode+f'-answers_seed_{args.seed}.json'), mode='w', encoding='utf-8') as f: json.dump(answers, f, indent=4, ensure_ascii=False)
    else:
        with open(os.path.join(args.output_dir, f'{model_name}_gpt-'+args.mode+f'-answers_seed_{args.seed}.json'), mode='w', encoding='utf-8') as f:
            json.dump(answers, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
