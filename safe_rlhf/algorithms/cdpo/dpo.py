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
from trl import DPOTrainer
import argparse
import wandb
import random
import numpy as np
from torch.utils.data import random_split

from pku_safe import PKU, PKU_C

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="PKU-Alignment/alpaca-7b-reproduced",
        metadata={"help": "the location of the SFT model name or path"},
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the number of warmup steps"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    gradient_checkpointing_use_reentrant: Optional[bool] = field(
        default=False, metadata={"help": "whether to use reentrant for gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=512, metadata={"help": "the maximum sequence length"})
    max_steps: Optional[int] = field(default=3000, metadata={"help": "max number of training steps"})
    logging_steps: Optional[int] = field(default=50, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=200, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=200, metadata={"help": "the evaluation frequency"})
    num_train_epochs: Optional[int] = field(default=3, metadata={"help": "the number of training epochs"})

    output_dir: Optional[str] = field(default="./results", metadata={"help": "the output directory"})
    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    load_in_4bit: Optional[bool] = field(default=True, metadata={"help": "whether to load the model in 4bit"})
    model_dtype: Optional[str] = field(
        default="bfloat16", metadata={"help": "model_dtype[float16, bfloat16, float] for loading."}
    )

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    lamb: float = field(default=None, metadata={"help": "the lambda parameter for DPO loss"})
    prefer_safe: bool = field(default=True, metadata={"help": "whether to prefer safe responses"})
    b_bar: float = field(default=0.0, metadata={"help": "the b_bar parameter for DPO loss"})

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
    
    train_dataset = dataset["train"]
    # split the train dataset into train and eval with 90% and 10% respectively
    ds = train_dataset.train_test_split(test_size=0.1,)
    train_dataset = ds['train']
    eval_dataset = ds['test']

    original_columns = train_dataset.column_names

    def return_prompt_and_responses(samples) -> Dict[str, str]:
            response_0 = samples["response_0"]
            response_1 = samples["response_1"] 
            responses_list = list(zip(response_0, response_1))
            if prefer_safe:
                indices = samples["safer_response_id"]
            else:
                indices = samples['better_response_id']
            return {
                "prompt": [PROMPT_BEGIN + PROMPT_USER.format(input=question) + PROMPT_ASSISTANT for question in samples["prompt"]],
                "chosen": [responses[safer_index] for responses, safer_index in zip(responses_list, indices)],
                "rejected": [responses[1-safer_index] for responses, safer_index in zip(responses_list, indices)]
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

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        entity="lishuo1-penn",
        project="rlhf-dpo",
        # track hyperparameters and run metadata
        config=script_args.__dict__,
        # turn off wandb
        # comment this line to turn on wandb
        # mode="disabled",
    )

     # 1. load a pretrained model
    torch_dtype = torch.float
    if script_args.model_dtype == "float16":
        torch_dtype = torch.float16
    elif script_args.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, save_in_4bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        device_map={"": Accelerator().local_process_index},
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    tokenizer = AutoTokenizer.from_pretrained("PKU-Alignment/alpaca-7b-reproduced")
    tokenizer.pad_token = tokenizer.eos_token
    
    set_seed(script_args.seed)
    if script_args.lamb is None:
        trainset, evalset = get_PKU(prefer_safe=script_args.prefer_safe)
    else:
        import json
        with open('PKU-SafeRLHF-30K_train_v3.json', 'r') as f:
            train_dataset = json.load(f)
        trainset = Dataset.from_dict(train_dataset)
        ds = trainset.train_test_split(test_size=0.1,)
        trainset = ds['train']
        evalset = ds['test']
        b_bar = script_args.b_bar

        def return_prompt_and_responses(samples):
            reward_scores_0 = torch.tensor(samples['reward_scores_0']).to(model.device)
            reward_scores_1 = torch.tensor(samples['reward_scores_1']).to(model.device)
            safety_scores_0 = -torch.tensor(samples['cost_scores_0']).to(model.device)
            safety_scores_1 = -torch.tensor(samples['cost_scores_1']).to(model.device)
            weight_0s = reward_scores_0 + script_args.lamb * safety_scores_0
            weight_1s = reward_scores_1 + script_args.lamb * safety_scores_1
            indices = []
            # for i in range(len(weight_0s)):
            #     if weight_0s[i] > weight_1s[i]:
            #         indices.append(0)
            #     else:
            #         indices.append(1)
            probs = torch.sigmoid(weight_1s - weight_0s)
            indices = torch.bernoulli(probs).long().squeeze()
            return {
                    "prompt": [tokenizer.decode(prompt, skip_special_tokens=True) for prompt in samples['prompt_ids']],
                    "chosen": [tokenizer.decode(samples["response_0_ids"][i], skip_special_tokens=True) if index == 0 else tokenizer.decode(samples["response_1_ids"][i], skip_special_tokens=True) for i, index in enumerate(indices)],
                    "rejected": [tokenizer.decode(samples["response_1_ids"][i], skip_special_tokens=True) if index == 0 else tokenizer.decode(samples["response_0_ids"][i], skip_special_tokens=True) for i, index in enumerate(indices)]
                }

        trainset = trainset.map(return_prompt_and_responses, batched=True)
        evalset = evalset.map(return_prompt_and_responses, batched=True)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        # max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        num_train_epochs=script_args.num_train_epochs,
        # evaluation_strategy='no',
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=True,
        remove_unused_columns=False,
        run_name="dpo_llama_pku",
        gradient_checkpointing_kwargs=dict(use_reentrant=script_args.gradient_checkpointing_use_reentrant),
        seed=script_args.seed,
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
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

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=trainset,
        eval_dataset=evalset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
